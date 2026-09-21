using FrontRunningLabeler.Cli;
using FrontRunningLabeler.Services;

namespace FrontRunningLabeler;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            var options = CommandLineOptions.Parse(args);
            if (options.ShowHelp)
            {
                Console.WriteLine(CommandLineOptions.Usage);
                return 0;
            }

            var datasets = InputDiscovery.Discover(options);
            Console.WriteLine($"Escopo: {options.InputMode}; detectores: {string.Join(", ", options.Detectors)}");
            Console.WriteLine($"Datasets selecionados: {datasets.Count}");

            var pipeline = new DetectionPipeline(options);
            var completed = 0;
            foreach (var dataset in datasets)
            {
                completed++;
                Console.WriteLine($"\n[{completed}/{datasets.Count}] {dataset.Configuration}/{dataset.Split}");
                pipeline.Run(dataset);
            }

            Console.WriteLine("\nProcessamento concluído.");
            return 0;
        }
        catch (CommandLineException exception)
        {
            Console.Error.WriteLine($"Erro de parâmetros: {exception.Message}\n");
            Console.Error.WriteLine(CommandLineOptions.Usage);
            return 2;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Falha: {exception.Message}");
            Console.Error.WriteLine(exception.StackTrace);
            return 1;
        }
    }
}
