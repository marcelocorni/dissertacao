using System.Globalization;
using System.Text.Json;
using DuckDB.NET.Data;
using FrontRunningLabeler.Cli;
using FrontRunningLabeler.Models;
using FrontRunningLabeler.Utils;

namespace FrontRunningLabeler.Services;

public sealed class DetectionPipeline(CommandLineOptions options)
{
    private const string ProjectVersion = "3.0.0-candidate-first";

    public void Run(DatasetDescriptor dataset)
    {
        var outputDirectory = Path.Combine(options.OutputDirectory, dataset.InputMode, dataset.Configuration, dataset.Split);
        var eventsPath = Path.Combine(outputDirectory, "01_detection_events.parquet");
        var labelsPath = Path.Combine(outputDirectory, "02_transaction_labels.parquet");
        var summaryPath = Path.Combine(outputDirectory, "03_summary.csv");
        var manifestPath = Path.Combine(outputDirectory, "04_manifest_run.json");
        var enrichmentQueuePath = Path.Combine(outputDirectory, "05_enrichment_queue.parquet");
        var artifacts = new[] { eventsPath, labelsPath, summaryPath, manifestPath, enrichmentQueuePath };

        if (artifacts.All(File.Exists) && !options.Overwrite)
        {
            Console.WriteLine("Artefatos completos já existem; dataset ignorado. Use --overwrite para reprocessar.");
            return;
        }
        if (artifacts.Any(File.Exists) && !options.Overwrite)
            throw new InvalidOperationException($"Há artefatos parciais em {outputDirectory}. Revise-os ou use --overwrite.");
        if (options.Overwrite)
            foreach (var artifact in artifacts.Where(File.Exists)) File.Delete(artifact);

        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(options.TempDirectory);
        using var connection = new DuckDBConnection("Data Source=:memory:");
        connection.Open();
        Execute(connection, $"SET threads={options.Threads}");
        Execute(connection, $"SET memory_limit={SqlText.Literal(options.MemoryLimit)}");
        Execute(connection, $"SET temp_directory={SqlText.PathLiteral(options.TempDirectory)}");
        Execute(connection, "SET preserve_insertion_order=false");

        CreateAnchors(connection, dataset);
        ValidateAnchors(connection, dataset);
        var dates = ReadDates(connection);
        var rawFiles = ResolveRawFiles(dates);
        Console.WriteLine($"Âncoras: {ScalarLong(connection, "SELECT count(*) FROM anchors"):N0}; datas: {dates.Count}; Parquets brutos: {rawFiles.Count}");

        if (options.DryRun)
        {
            Console.WriteLine("Dry-run concluído: entrada, esquema, datas e partições brutas validados.");
            return;
        }

        Console.WriteLine("Materializando somente as transações dos blocos relevantes...");
        CreateRelevantTransactions(connection, rawFiles);
        ValidateRawCoverage(connection);
        CreateEventsTable(connection);

        foreach (var detector in options.Detectors)
        {
            Console.WriteLine($"Executando detector: {detector}");
            switch (detector)
            {
                case "insertion": DetectInsertion(connection, dataset); break;
                case "displacement": DetectDisplacement(connection, dataset); break;
                case "suppression": DetectSuppression(connection, dataset); break;
                default: throw new InvalidOperationException($"Detector não implementado: {detector}");
            }
        }

        var anchors = ScalarLong(connection, "SELECT count(*) FROM anchors");
        var blocks = ScalarLong(connection, "SELECT count(*) FROM relevant_blocks");
        var transactions = ScalarLong(connection, "SELECT count(*) FROM block_transactions");
        var events = ScalarLong(connection, "SELECT count(*) FROM detection_events");
        var confirmedEvents = ScalarLong(connection, "SELECT count(*) FROM detection_events WHERE validation_status = 'confirmed'");
        var patternAnchors = ScalarLong(connection, "SELECT count(DISTINCT candidate_hash) FROM detection_events");
        var candidateAttackers = ScalarLong(connection, """
            SELECT count(DISTINCT hash) FROM (
                SELECT attacker_front_hash AS hash FROM detection_events
                UNION ALL
                SELECT attacker_back_hash AS hash FROM detection_events WHERE attacker_back_hash IS NOT NULL
            )
            """);
        var victims = ScalarLong(connection, "SELECT count(DISTINCT victim_hash) FROM detection_events WHERE victim_hash IS NOT NULL");

        ExportEvents(connection, eventsPath);
        ExportLabels(connection, labelsPath, dataset);
        ExportSummary(connection, summaryPath);
        ExportEnrichmentQueue(connection, enrichmentQueuePath);
        var mappedPositives = ScalarLong(connection, $"SELECT sum(label)::BIGINT FROM read_parquet({SqlText.PathLiteral(labelsPath)})");
        var enrichmentTransactions = ScalarLong(connection, $"SELECT count(*) FROM read_parquet({SqlText.PathLiteral(enrichmentQueuePath)})");

        var manifest = new RunManifest(
            DateTimeOffset.UtcNow,
            ProjectVersion,
            dataset.InputMode,
            dataset.InputPath,
            dataset.Configuration,
            dataset.Split,
            options.Detectors.ToArray(),
            options.RawRootTemplate,
            options.ConfigurationFile,
            dates.Count,
            rawFiles.Count,
            anchors,
            blocks,
            transactions,
            events,
            confirmedEvents,
            patternAnchors,
            mappedPositives,
            candidateAttackers,
            victims,
            enrichmentTransactions,
            eventsPath,
            labelsPath,
            summaryPath,
            enrichmentQueuePath,
            true,
            "A versão 3 gera candidatos e não promove heurísticas estruturais diretamente a ground truth. " +
            "label=1 exige validation_status=confirmed, definido somente após validação semântica por receipts/logs. " +
            (dataset.InputMode == "sampled"
                ? "O modo sampled é independente da seleção AE/IF e é o recomendado para avaliação."
                : "O modo candidates reproduz o escopo condicionado às anomalias e não deve ser usado sozinho para ROC/AUC."));
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"Candidatos: {events:N0}; confirmados: {confirmedEvents:N0}; âncoras com padrão: {patternAnchors:N0}; atacantes candidatos: {candidateAttackers:N0}; fila de enriquecimento: {enrichmentTransactions:N0}");
        Console.WriteLine($"Saída: {outputDirectory}");
    }

    private static void CreateAnchors(DuckDBConnection connection, DatasetDescriptor dataset)
    {
        var sourceColumns = dataset.InputMode == "candidates"
            ? "coalesce(detection_group, 'candidate')::VARCHAR AS model_detection_source"
            : "'independent_sampled'::VARCHAR AS model_detection_source";
        Execute(connection, $"""
            CREATE TEMP TABLE anchors AS
            SELECT
                row_id::UBIGINT AS row_id,
                lower(hash)::VARCHAR AS hash,
                block_number::BIGINT AS block_number,
                cast(date AS VARCHAR) AS date,
                lower(from_address)::VARCHAR AS from_address,
                lower(to_address)::VARCHAR AS to_address,
                transaction_index::BIGINT AS transaction_index,
                transaction_type::BIGINT AS transaction_type,
                {sourceColumns}
            FROM read_parquet({SqlText.PathLiteral(dataset.InputPath)})
            """);
        Execute(connection, "CREATE TEMP TABLE relevant_blocks AS SELECT DISTINCT block_number FROM anchors");
    }

    private static void ValidateAnchors(DuckDBConnection connection, DatasetDescriptor dataset)
    {
        var stats = ReadLongs(connection, """
            SELECT count(*), count(DISTINCT row_id), count(DISTINCT hash),
                   count(*) FILTER (WHERE hash IS NULL OR block_number IS NULL OR date IS NULL)
            FROM anchors
            """);
        if (stats[0] == 0) throw new InvalidOperationException($"Entrada vazia: {dataset.InputPath}");
        if (stats[0] != stats[1] || stats[0] != stats[2])
            throw new InvalidOperationException("row_id e hash precisam ser únicos dentro do recorte.");
        if (stats[3] != 0) throw new InvalidOperationException("Há chaves obrigatórias nulas na entrada.");
    }

    private static List<DateOnly> ReadDates(DuckDBConnection connection)
    {
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT DISTINCT date FROM anchors ORDER BY date";
        using var reader = command.ExecuteReader();
        var dates = new List<DateOnly>();
        while (reader.Read())
        {
            var value = reader.GetString(0);
            if (!DateOnly.TryParseExact(value, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var date))
                throw new InvalidOperationException($"Data inválida na entrada: {value}");
            dates.Add(date);
        }
        return dates;
    }

    private List<string> ResolveRawFiles(IEnumerable<DateOnly> dates)
    {
        var files = new List<string>();
        foreach (var date in dates)
        {
            var root = options.RawRootTemplate.Replace("{year}", date.Year.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal);
            var partition = Path.Combine(root, $"date={date:yyyy-MM-dd}");
            if (!Directory.Exists(partition))
                throw new DirectoryNotFoundException($"Partição bruta ausente: {partition}");
            var daily = Directory.EnumerateFiles(partition, "*.parquet", SearchOption.TopDirectoryOnly).Order().ToArray();
            if (daily.Length == 0) throw new FileNotFoundException($"Nenhum Parquet em {partition}");
            files.AddRange(daily);
        }
        return files;
    }

    private static void CreateRelevantTransactions(DuckDBConnection connection, IReadOnlyCollection<string> rawFiles)
    {
        Execute(connection, $"""
            CREATE TEMP TABLE block_transactions AS
            SELECT
                lower(r.hash)::VARCHAR AS hash,
                r.block_number::BIGINT AS block_number,
                r.block_timestamp::TIMESTAMP AS block_timestamp,
                lower(r.from_address)::VARCHAR AS from_address,
                lower(r.to_address)::VARCHAR AS to_address,
                r.transaction_index::BIGINT AS transaction_index,
                r.gas_price::DOUBLE AS gas_price,
                r.value::DOUBLE AS value
            FROM read_parquet({SqlText.PathList(rawFiles)}, union_by_name=true, hive_partitioning=false) r
            SEMI JOIN relevant_blocks b USING (block_number)
            """);
        Execute(connection, "CREATE INDEX idx_transactions_hash ON block_transactions(hash)");
        Execute(connection, "CREATE INDEX idx_transactions_block_index ON block_transactions(block_number, transaction_index)");
    }

    private static void ValidateRawCoverage(DuckDBConnection connection)
    {
        var missing = ScalarLong(connection, "SELECT count(*) FROM anchors a ANTI JOIN block_transactions t USING (hash)");
        if (missing != 0)
            throw new InvalidOperationException($"{missing:N0} âncoras não foram encontradas nos Parquets brutos.");
        var duplicatePositions = ScalarLong(connection, """
            SELECT count(*) FROM (
                SELECT block_number, transaction_index
                FROM block_transactions
                GROUP BY ALL HAVING count(*) > 1
            )
            """);
        if (duplicatePositions != 0)
            throw new InvalidOperationException($"Foram encontradas {duplicatePositions:N0} posições duplicadas nos blocos.");
    }

    private static void CreateEventsTable(DuckDBConnection connection) => Execute(connection, """
        CREATE TEMP TABLE detection_events (
            dataset_id VARCHAR,
            input_mode VARCHAR,
            configuration VARCHAR,
            split VARCHAR,
            detector VARCHAR,
            candidate_row_id UBIGINT,
            candidate_hash VARCHAR,
            candidate_role VARCHAR,
            attacker_hash VARCHAR,
            attacker_front_hash VARCHAR,
            attacker_back_hash VARCHAR,
            victim_hash VARCHAR,
            candidate_from_address VARCHAR,
            candidate_to_address VARCHAR,
            block_number BIGINT,
            block_timestamp TIMESTAMP,
            candidate_index BIGINT,
            model_detection_source VARCHAR,
            related_hash_1 VARCHAR,
            related_role_1 VARCHAR,
            related_index_1 BIGINT,
            related_hash_2 VARCHAR,
            related_role_2 VARCHAR,
            related_index_2 BIGINT,
            rule_version VARCHAR,
            rule_description VARCHAR,
            validation_status VARCHAR,
            semantic_requirements VARCHAR
        )
        """);

    private static void DetectInsertion(DuckDBConnection connection, DatasetDescriptor dataset) => Execute(connection, $"""
        INSERT INTO detection_events
        SELECT
            {SqlText.Literal(dataset.Id)}, {SqlText.Literal(dataset.InputMode)},
            {SqlText.Literal(dataset.Configuration)}, {SqlText.Literal(dataset.Split)},
            'insertion', a.row_id, a.hash, 'victim',
            previous_tx.hash, previous_tx.hash, next_tx.hash, current_tx.hash,
            current_tx.from_address, current_tx.to_address, current_tx.block_number,
            current_tx.block_timestamp, current_tx.transaction_index, a.model_detection_source,
            previous_tx.hash, 'attacker_front_candidate', previous_tx.transaction_index,
            next_tx.hash, 'attacker_back_candidate', next_tx.transaction_index,
            'insertion-v3.0-candidate',
            'As transações externas têm remetente, destinatário e valor equivalentes; a transação intermediária é a vítima candidata.',
            'pending_semantic_validation',
            'Confirmar mesmo pool; front e vítima na mesma direção; back na direção oposta; calcular resultado econômico das pernas externas.'
        FROM anchors a
        JOIN block_transactions current_tx ON current_tx.hash = a.hash
        JOIN block_transactions previous_tx
          ON previous_tx.block_number = current_tx.block_number
         AND previous_tx.transaction_index = current_tx.transaction_index - 1
        JOIN block_transactions next_tx
          ON next_tx.block_number = current_tx.block_number
         AND next_tx.transaction_index = current_tx.transaction_index + 1
        WHERE current_tx.transaction_index >= 1
          AND previous_tx.from_address = next_tx.from_address
          AND previous_tx.to_address = next_tx.to_address
          AND previous_tx.value IS NOT NULL AND next_tx.value IS NOT NULL
          AND abs(previous_tx.value - next_tx.value) <= greatest(abs(previous_tx.value), 1.0) * 1e-12
          AND coalesce(current_tx.value, 0) = 0
          AND previous_tx.from_address <> current_tx.from_address
        """);

    private static void DetectDisplacement(DuckDBConnection connection, DatasetDescriptor dataset) => Execute(connection, $"""
        INSERT INTO detection_events
        WITH ordered_candidates AS (
            -- A âncora é a vítima; a transação imediatamente anterior é o atacante candidato.
            SELECT
                a.row_id AS candidate_row_id, a.hash AS candidate_hash, 'victim'::VARCHAR AS candidate_role,
                previous_tx.hash AS attacker_hash, previous_tx.hash AS attacker_front_hash,
                NULL::VARCHAR AS attacker_back_hash, current_tx.hash AS victim_hash,
                current_tx.from_address AS candidate_from_address,
                current_tx.to_address AS candidate_to_address,
                current_tx.block_number, current_tx.block_timestamp,
                current_tx.transaction_index AS candidate_index, a.model_detection_source,
                previous_tx.hash AS related_hash_1,
                'displacer_attacker_candidate'::VARCHAR AS related_role_1,
                previous_tx.transaction_index AS related_index_1,
                current_tx.hash AS related_hash_2, 'victim_candidate'::VARCHAR AS related_role_2,
                current_tx.transaction_index AS related_index_2
            FROM anchors a
            JOIN block_transactions current_tx ON current_tx.hash = a.hash
            JOIN block_transactions previous_tx
              ON previous_tx.block_number = current_tx.block_number
             AND previous_tx.transaction_index = current_tx.transaction_index - 1
            WHERE current_tx.transaction_index >= 1
              AND previous_tx.gas_price IS NOT NULL AND current_tx.gas_price IS NOT NULL
              AND previous_tx.gas_price > current_tx.gas_price
              AND previous_tx.to_address = current_tx.to_address
              AND previous_tx.from_address <> current_tx.from_address
              AND abs(coalesce(previous_tx.value, 0) - coalesce(current_tx.value, 0))
                  <= greatest(abs(coalesce(current_tx.value, 0)), 1.0) * 0.01

            UNION ALL

            -- A âncora é o atacante; a transação imediatamente seguinte é a vítima candidata.
            SELECT
                a.row_id AS candidate_row_id, a.hash, 'attacker_front', current_tx.hash, current_tx.hash,
                NULL::VARCHAR, next_tx.hash, current_tx.from_address, current_tx.to_address,
                current_tx.block_number, current_tx.block_timestamp,
                current_tx.transaction_index, a.model_detection_source,
                current_tx.hash, 'displacer_attacker_candidate', current_tx.transaction_index,
                next_tx.hash, 'victim_candidate', next_tx.transaction_index
            FROM anchors a
            JOIN block_transactions current_tx ON current_tx.hash = a.hash
            JOIN block_transactions next_tx
              ON next_tx.block_number = current_tx.block_number
             AND next_tx.transaction_index = current_tx.transaction_index + 1
            WHERE current_tx.gas_price IS NOT NULL AND next_tx.gas_price IS NOT NULL
              AND current_tx.gas_price > next_tx.gas_price
              AND current_tx.to_address = next_tx.to_address
              AND current_tx.from_address <> next_tx.from_address
              AND abs(coalesce(current_tx.value, 0) - coalesce(next_tx.value, 0))
                  <= greatest(abs(coalesce(next_tx.value, 0)), 1.0) * 0.01
        ), deduplicated AS (
            SELECT *, row_number() OVER (
                PARTITION BY attacker_front_hash, victim_hash
                ORDER BY candidate_row_id
            ) AS occurrence
            FROM ordered_candidates
        )
        SELECT
            {SqlText.Literal(dataset.Id)}, {SqlText.Literal(dataset.InputMode)},
            {SqlText.Literal(dataset.Configuration)}, {SqlText.Literal(dataset.Split)},
            'displacement', candidate_row_id, candidate_hash, candidate_role,
            attacker_hash, attacker_front_hash, attacker_back_hash, victim_hash,
            candidate_from_address, candidate_to_address, block_number, block_timestamp,
            candidate_index, model_detection_source,
            related_hash_1, related_role_1, related_index_1,
            related_hash_2, related_role_2, related_index_2,
            'displacement-v3.0-candidate',
            'O atacante candidato aparece imediatamente antes da vítima, usa o mesmo destino, remetente distinto, valor semelhante e gas price superior.',
            'pending_semantic_validation',
            'Confirmar mesma intenção econômica, seletor, parâmetros, protocolo/pool e efeito adverso sobre a vítima.'
        FROM deduplicated WHERE occurrence = 1
        """);

    private static void DetectSuppression(DuckDBConnection connection, DatasetDescriptor dataset) => Execute(connection, $"""
        INSERT INTO detection_events
        WITH qualifying AS (
            SELECT
                a.row_id, a.hash AS candidate_hash, a.model_detection_source,
                current_tx.from_address AS candidate_from, current_tx.to_address AS candidate_to,
                current_tx.block_number, current_tx.block_timestamp,
                current_tx.transaction_index AS candidate_index,
                attacker.hash AS attacker_hash, attacker.transaction_index AS attacker_index,
                row_number() OVER (PARTITION BY a.hash ORDER BY attacker.transaction_index DESC) AS attack_order,
                count(*) OVER (PARTITION BY a.hash) AS attack_count
            FROM anchors a
            JOIN block_transactions current_tx ON current_tx.hash = a.hash
            JOIN block_transactions attacker
              ON attacker.block_number = current_tx.block_number
             AND attacker.transaction_index < current_tx.transaction_index
             AND attacker.transaction_index >= greatest(current_tx.transaction_index - 5, 0)
            WHERE attacker.from_address <> current_tx.from_address
              AND attacker.to_address = current_tx.to_address
              AND attacker.value > 0
              AND attacker.gas_price > current_tx.gas_price
        ), grouped AS (
            SELECT
                row_id, candidate_hash, model_detection_source, candidate_from,
                candidate_to, block_number, block_timestamp, candidate_index,
                max(CASE WHEN attack_order = 1 THEN attacker_hash END) AS attacker_1,
                max(CASE WHEN attack_order = 1 THEN attacker_index END) AS attacker_index_1,
                max(CASE WHEN attack_order = 2 THEN attacker_hash END) AS attacker_2,
                max(CASE WHEN attack_order = 2 THEN attacker_index END) AS attacker_index_2
            FROM qualifying
            WHERE attack_count >= 2 AND attack_order <= 2
            GROUP BY ALL
        )
        SELECT
            {SqlText.Literal(dataset.Id)}, {SqlText.Literal(dataset.InputMode)},
            {SqlText.Literal(dataset.Configuration)}, {SqlText.Literal(dataset.Split)},
            'suppression', row_id, candidate_hash, 'victim',
            attacker_1, attacker_1, attacker_2, candidate_hash,
            candidate_from, candidate_to, block_number, block_timestamp, candidate_index,
            model_detection_source,
            attacker_1, 'suppression_preceding_candidate_1', attacker_index_1,
            attacker_2, 'suppression_preceding_candidate_2', attacker_index_2,
            'suppression-v3.0-heuristic',
            'Ao menos duas transações nas cinco posições anteriores têm remetente distinto, mesmo destino, valor positivo e gas price superior.',
            'requires_mempool_evidence',
            'A blockchain confirmada não prova exclusão ou atraso; exigir observação de mempool antes de usar como ground truth.'
        FROM grouped
        """);

    private static void ExportEvents(DuckDBConnection connection, string path) => Execute(connection, $"""
        COPY (
            SELECT row_number() OVER (ORDER BY split, block_number, candidate_index, detector, attacker_hash)::UBIGINT AS detection_event_id, *
            FROM detection_events
        ) TO {SqlText.PathLiteral(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """);

    private static void ExportLabels(DuckDBConnection connection, string path, DatasetDescriptor dataset) => Execute(connection, $"""
        COPY (
            WITH confirmed_attacker_events AS (
                SELECT attacker_front_hash AS hash,
                       detector
                FROM detection_events
                WHERE validation_status = 'confirmed'
                UNION ALL
                SELECT attacker_back_hash AS hash, detector
                FROM detection_events
                WHERE validation_status = 'confirmed' AND attacker_back_hash IS NOT NULL
            ), attacker_labels AS (
                SELECT hash,
                       string_agg(DISTINCT detector, '|' ORDER BY detector) AS front_running_types
                FROM confirmed_attacker_events GROUP BY hash
            ), candidate_attacker_labels AS (
                SELECT hash,
                       string_agg(DISTINCT detector, '|' ORDER BY detector) AS candidate_front_running_types
                FROM (
                    SELECT attacker_front_hash AS hash, detector FROM detection_events
                    UNION ALL
                    SELECT attacker_back_hash AS hash, detector FROM detection_events
                    WHERE attacker_back_hash IS NOT NULL
                )
                GROUP BY hash
            ),
            anchor_labels AS (
                SELECT candidate_hash AS hash,
                       string_agg(DISTINCT detector, '|' ORDER BY detector) AS anchor_pattern_types
                FROM detection_events GROUP BY candidate_hash
            ),
            victim_labels AS (
                SELECT victim_hash AS hash,
                       string_agg(DISTINCT detector, '|' ORDER BY detector) AS victim_pattern_types
                FROM detection_events
                WHERE victim_hash IS NOT NULL AND validation_status = 'confirmed'
                GROUP BY victim_hash
            ), candidate_victim_labels AS (
                SELECT victim_hash AS hash,
                       string_agg(DISTINCT detector, '|' ORDER BY detector) AS candidate_victim_pattern_types
                FROM detection_events WHERE victim_hash IS NOT NULL GROUP BY victim_hash
            )
            SELECT
                {SqlText.Literal("detector_csharp_semantic_v3")}::VARCHAR AS label_source,
                {SqlText.Literal(dataset.InputMode)}::VARCHAR AS input_mode,
                {SqlText.Literal(dataset.Configuration)}::VARCHAR AS configuration,
                {SqlText.Literal(dataset.Split)}::VARCHAR AS split,
                a.model_detection_source,
                a.row_id,
                a.hash,
                a.block_number,
                a.date,
                a.from_address,
                a.to_address,
                a.transaction_index,
                CASE WHEN attacker.hash IS NOT NULL THEN 1 ELSE 0 END::UTINYINT AS label,
                attacker.hash IS NOT NULL AS is_attacker_transaction,
                candidate_attacker.hash IS NOT NULL AS is_candidate_attacker,
                anchor.hash IS NOT NULL AS is_pattern_anchor,
                victim.hash IS NOT NULL AS is_victim_transaction,
                candidate_victim.hash IS NOT NULL AS is_candidate_victim,
                attacker.front_running_types,
                candidate_attacker.candidate_front_running_types,
                anchor.anchor_pattern_types,
                victim.victim_pattern_types,
                candidate_victim.candidate_victim_pattern_types
            FROM anchors a
            LEFT JOIN attacker_labels attacker USING (hash)
            LEFT JOIN candidate_attacker_labels candidate_attacker USING (hash)
            LEFT JOIN anchor_labels anchor USING (hash)
            LEFT JOIN victim_labels victim USING (hash)
            LEFT JOIN candidate_victim_labels candidate_victim USING (hash)
        ) TO {SqlText.PathLiteral(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """);

    private static void ExportSummary(DuckDBConnection connection, string path) => Execute(connection, $"""
        COPY (
            WITH attacker_hashes AS (
                SELECT detector, attacker_front_hash AS hash FROM detection_events
                UNION ALL
                SELECT detector, attacker_back_hash AS hash FROM detection_events
                WHERE attacker_back_hash IS NOT NULL
            ), per_detector AS (
                SELECT detector, count(DISTINCT hash)::BIGINT AS candidate_attacker_transactions
                FROM attacker_hashes GROUP BY detector
            )
            SELECT e.detector, count(*)::BIGINT AS candidate_events,
                   count(*) FILTER (WHERE validation_status = 'confirmed')::BIGINT AS confirmed_events,
                   count(DISTINCT candidate_hash)::BIGINT AS pattern_anchors,
                   p.candidate_attacker_transactions,
                   count(DISTINCT victim_hash)::BIGINT AS victim_transactions
            FROM detection_events e JOIN per_detector p USING (detector)
            GROUP BY e.detector, p.candidate_attacker_transactions
            UNION ALL
            SELECT '__total__', count(*)::BIGINT,
                   count(*) FILTER (WHERE validation_status = 'confirmed')::BIGINT,
                   count(DISTINCT candidate_hash)::BIGINT,
                   (SELECT count(DISTINCT hash) FROM attacker_hashes)::BIGINT,
                   count(DISTINCT victim_hash)::BIGINT
            FROM detection_events
            ORDER BY detector
        ) TO {SqlText.PathLiteral(path)} (HEADER, DELIMITER ',')
        """);

    private static void ExportEnrichmentQueue(DuckDBConnection connection, string path) => Execute(connection, $"""
        COPY (
            WITH requested AS (
                SELECT dataset_id, split, detector, attacker_front_hash AS hash, 'attacker_front' AS role
                FROM detection_events WHERE detector <> 'suppression'
                UNION ALL
                SELECT dataset_id, split, detector, attacker_back_hash AS hash, 'attacker_back' AS role
                FROM detection_events
                WHERE detector <> 'suppression' AND attacker_back_hash IS NOT NULL
                UNION ALL
                SELECT dataset_id, split, detector, victim_hash AS hash, 'victim' AS role
                FROM detection_events
                WHERE detector <> 'suppression' AND victim_hash IS NOT NULL
            )
            SELECT
                r.dataset_id,
                r.split,
                r.hash,
                string_agg(DISTINCT r.detector, '|' ORDER BY r.detector) AS candidate_detectors,
                string_agg(DISTINCT r.role, '|' ORDER BY r.role) AS candidate_roles,
                t.block_number,
                t.block_timestamp,
                t.transaction_index,
                t.from_address,
                t.to_address,
                t.gas_price,
                t.value
            FROM requested r
            JOIN block_transactions t USING (hash)
            GROUP BY r.dataset_id, r.split, r.hash, t.block_number, t.block_timestamp,
                     t.transaction_index, t.from_address, t.to_address, t.gas_price, t.value
            ORDER BY t.block_number, t.transaction_index, r.hash
        ) TO {SqlText.PathLiteral(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """);

    private static void Execute(DuckDBConnection connection, string sql)
    {
        using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.ExecuteNonQuery();
    }

    private static long ScalarLong(DuckDBConnection connection, string sql)
    {
        using var command = connection.CreateCommand();
        command.CommandText = sql;
        var value = command.ExecuteScalar();
        return value is null or DBNull ? 0 : Convert.ToInt64(value, CultureInfo.InvariantCulture);
    }

    private static long[] ReadLongs(DuckDBConnection connection, string sql)
    {
        using var command = connection.CreateCommand();
        command.CommandText = sql;
        using var reader = command.ExecuteReader();
        if (!reader.Read()) throw new InvalidOperationException("Consulta de validação não retornou resultado.");
        return Enumerable.Range(0, reader.FieldCount)
            .Select(index => Convert.ToInt64(reader.GetValue(index), CultureInfo.InvariantCulture)).ToArray();
    }
}
