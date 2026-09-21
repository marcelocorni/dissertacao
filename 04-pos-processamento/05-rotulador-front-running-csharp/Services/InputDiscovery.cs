using System.Text.RegularExpressions;
using FrontRunningLabeler.Cli;
using FrontRunningLabeler.Models;

namespace FrontRunningLabeler.Services;

public static partial class InputDiscovery
{
    public static IReadOnlyList<DatasetDescriptor> Discover(CommandLineOptions options)
    {
        var files = File.Exists(options.InputPath)
            ? [options.InputPath]
            : options.InputMode == "sampled"
                ? Directory.EnumerateFiles(options.InputPath, "05_metadados.parquet", SearchOption.AllDirectories).ToArray()
                : Directory.EnumerateFiles(options.InputPath, "*.parquet", SearchOption.AllDirectories).ToArray();

        var datasets = files.Select(path => Describe(path, options.InputMode))
            .Where(item => options.ConfigurationFilter is null || item.Configuration.Equals(options.ConfigurationFilter, StringComparison.OrdinalIgnoreCase))
            .Where(item => options.SplitFilter is null || item.Split.Equals(options.SplitFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(item => item.Configuration, StringComparer.Ordinal)
            .ThenBy(item => item.Split, StringComparer.Ordinal)
            .ToArray();

        if (datasets.Length == 0)
            throw new CommandLineException("Nenhum dataset corresponde aos filtros informados.");
        var duplicate = datasets.GroupBy(item => item.Id).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
            throw new CommandLineException($"Dataset duplicado na entrada: {duplicate.Key}");
        return datasets;
    }

    private static DatasetDescriptor Describe(string path, string mode)
    {
        var file = new FileInfo(path);
        if (mode == "candidates")
        {
            var candidateMatch = CandidateFileName().Match(file.Name);
            if (!candidateMatch.Success)
                throw new CommandLineException($"Nome de candidato inválido: {file.Name}");
            return new DatasetDescriptor(file.FullName, mode, file.Directory!.Name, candidateMatch.Groups[1].Value);
        }

        if (!file.Name.Equals("05_metadados.parquet", StringComparison.OrdinalIgnoreCase))
            throw new CommandLineException("No modo sampled, o arquivo precisa se chamar 05_metadados.parquet.");
        var parent = file.Directory!.Name;
        var split = parent.Equals("resultados-matrizes", StringComparison.OrdinalIgnoreCase)
            ? "treino_2024_pos_dencun"
            : SplitDirectory().Match(parent) is { Success: true } splitMatch
                ? splitMatch.Groups[1].Value
                : throw new CommandLineException($"Não foi possível inferir o recorte a partir de {parent}.");
        return new DatasetDescriptor(file.FullName, mode, "independent_sampled", split);
    }

    [GeneratedRegex(@"^\d+_(.+)\.parquet$", RegexOptions.IgnoreCase)]
    private static partial Regex CandidateFileName();

    [GeneratedRegex(@"^\d+-(.+)$", RegexOptions.IgnoreCase)]
    private static partial Regex SplitDirectory();
}
