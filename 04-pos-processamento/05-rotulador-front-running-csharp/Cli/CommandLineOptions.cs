using System.Text.Json;

namespace FrontRunningLabeler.Cli;

public sealed class CommandLineException(string message) : Exception(message);

public sealed record CommandLineOptions(
    string InputPath,
    string InputMode,
    string RawRootTemplate,
    string ConfigurationFile,
    string OutputDirectory,
    string TempDirectory,
    string MemoryLimit,
    int Threads,
    IReadOnlyList<string> Detectors,
    string? ConfigurationFilter,
    string? SplitFilter,
    bool Overwrite,
    bool DryRun,
    bool ShowHelp)
{
    private static readonly string[] SupportedDetectors = ["insertion", "displacement", "suppression"];

    public static string Usage => """
        Gerador de candidatos a front-running em Parquets Ethereum

        Uso:
          dotnet run -- --input <arquivo-ou-diretorio> [opções]

        Opções obrigatórias:
          --input <caminho>                 Metadados amostrados ou candidatos AE/IF.

        Opções:
          --input-mode sampled|candidates   Escopo de entrada. Padrão: sampled.
          --config <arquivo>                Padrão: .\configuracao\pipeline.json
          --raw-root-template <modelo>      Substitui dados.root_template nesta execução.
          --output-dir <diretório>          Padrão: .\resultados-v3
          --detectors <lista|all>           insertion,displacement,suppression. Padrão: all.
          --configuration <nome>            Filtra uma configuração no modo candidates.
          --split <nome>                    Filtra um recorte temporal.
          --temp-dir <diretório>            Padrão: src\.tmp\duckdb
          --memory-limit <valor>            Padrão: 12GB
          --threads <n>                     Padrão: quantidade lógica, máximo 8.
          --overwrite                       Sobrescreve somente os artefatos do dataset selecionado.
          --dry-run                         Valida entrada e partições sem executar as regras.
          --help                            Exibe esta ajuda.

        Observação:
          Esta etapa não produz rótulos positivos definitivos. Ela gera candidatos
          ordenados e uma fila de hashes para validação semântica por receipts/logs.

        Exemplos:
          dotnet run -- --input "..\..\03-machine-learning\01-features" --input-mode sampled --split teste_final_2025
          dotnet run -- --input "..\..\05-analise-resultados\resultados\candidatos" --input-mode candidates --configuration b_completa --detectors all
        """;

    public static CommandLineOptions Parse(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help", StringComparer.OrdinalIgnoreCase))
        {
            return new CommandLineOptions("", "sampled", "", "", "", "", "", 1, [], null, null, false, false, true);
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var switches = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < args.Length; index++)
        {
            var token = args[index];
            if (!token.StartsWith("--", StringComparison.Ordinal))
                throw new CommandLineException($"Argumento inesperado: {token}");

            if (token is "--overwrite" or "--dry-run")
            {
                switches.Add(token);
                continue;
            }

            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
                throw new CommandLineException($"Falta um valor para {token}.");
            values[token] = args[++index];
        }

        if (!values.TryGetValue("--input", out var input) || string.IsNullOrWhiteSpace(input))
            throw new CommandLineException("--input é obrigatório.");

        var mode = Get(values, "--input-mode", "sampled").ToLowerInvariant();
        if (mode is not ("sampled" or "candidates"))
            throw new CommandLineException("--input-mode deve ser sampled ou candidates.");

        var detectorText = Get(values, "--detectors", "all");
        var detectors = detectorText.Equals("all", StringComparison.OrdinalIgnoreCase)
            ? SupportedDetectors
            : detectorText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(item => item.ToLowerInvariant()).Distinct().ToArray();
        if (detectors.Length == 0 || detectors.Any(item => !SupportedDetectors.Contains(item)))
            throw new CommandLineException("--detectors contém um valor inválido.");

        var defaultThreads = Math.Min(Environment.ProcessorCount, 8);
        if (!int.TryParse(Get(values, "--threads", defaultThreads.ToString()), out var threads) || threads < 1)
            throw new CommandLineException("--threads deve ser um inteiro positivo.");

        var inputPath = Path.GetFullPath(input);
        if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
            throw new CommandLineException($"Entrada não encontrada: {inputPath}");

        var configPath = Path.GetFullPath(Get(values, "--config", FindDefaultConfigPath()));
        var configuredRawRoot = LoadRawRootTemplate(configPath);

        return new CommandLineOptions(
            inputPath,
            mode,
            Get(values, "--raw-root-template", configuredRawRoot),
            configPath,
            Path.GetFullPath(Get(values, "--output-dir", Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "resultados-v3"))),
            Path.GetFullPath(Get(values, "--temp-dir", Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", ".tmp", "duckdb"))),
            Get(values, "--memory-limit", "12GB"),
            threads,
            detectors,
            values.GetValueOrDefault("--configuration"),
            values.GetValueOrDefault("--split"),
            switches.Contains("--overwrite"),
            switches.Contains("--dry-run"),
            false);
    }

    private static string Get(IReadOnlyDictionary<string, string> values, string key, string fallback) =>
        values.TryGetValue(key, out var value) ? value : fallback;

    private static string FindDefaultConfigPath() =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "configuracao", "pipeline.json"));

    private static string LoadRawRootTemplate(string configPath)
    {
        if (!File.Exists(configPath))
            throw new CommandLineException($"Arquivo de configuração não encontrado: {configPath}");
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(configPath));
            var value = document.RootElement
                .GetProperty("dados")
                .GetProperty("root_template")
                .GetString();
            if (string.IsNullOrWhiteSpace(value) || !value.Contains("{year}", StringComparison.Ordinal))
                throw new CommandLineException(
                    $"dados.root_template deve ser um texto contendo {{year}} em {configPath}");
            return value;
        }
        catch (CommandLineException)
        {
            throw;
        }
        catch (Exception exception) when (exception is JsonException or InvalidOperationException or KeyNotFoundException)
        {
            throw new CommandLineException(
                $"Configuração inválida em {configPath}: esperado dados.root_template.");
        }
    }
}
