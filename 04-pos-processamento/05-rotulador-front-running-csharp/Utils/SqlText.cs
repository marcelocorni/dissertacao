namespace FrontRunningLabeler.Utils;

public static class SqlText
{
    public static string Literal(string value) => $"'{value.Replace("'", "''")}'";

    public static string PathLiteral(string path) => Literal(Path.GetFullPath(path).Replace('\\', '/'));

    public static string PathList(IEnumerable<string> paths) =>
        "[" + string.Join(",", paths.Select(PathLiteral)) + "]";
}
