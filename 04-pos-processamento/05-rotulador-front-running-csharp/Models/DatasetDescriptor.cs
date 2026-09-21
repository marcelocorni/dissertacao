namespace FrontRunningLabeler.Models;

public sealed record DatasetDescriptor(
    string InputPath,
    string InputMode,
    string Configuration,
    string Split)
{
    public string Id => $"{Configuration}__{Split}";
}

public sealed record RunManifest(
    DateTimeOffset CreatedAtUtc,
    string ProjectVersion,
    string InputMode,
    string InputFile,
    string Configuration,
    string Split,
    string[] Detectors,
    string RawRootTemplate,
    string ConfigurationFile,
    int SourceDates,
    int SourceParquetFiles,
    long Anchors,
    long RelevantBlocks,
    long RelevantTransactions,
    long CandidateEvents,
    long ConfirmedEvents,
    long PatternAnchors,
    long PositiveTransactions,
    long CandidateAttackerTransactions,
    long VictimTransactions,
    long EnrichmentTransactions,
    string EventsFile,
    string LabelsFile,
    string SummaryFile,
    string EnrichmentQueueFile,
    bool RequiresSemanticValidation,
    string MethodologicalNote);
