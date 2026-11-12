param(
    [Parameter(Mandatory = $true)][string]$PackageName,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Source
)

$ErrorActionPreference = "Stop"
$packageId = $PackageName.ToLowerInvariant()
$packageVersion = $Version.ToLowerInvariant()
$feedBase = $Source -replace '/index\.json$', ''
$packageFile = "$packageId.$packageVersion.nupkg"
$downloadDirectory = Join-Path $env:RUNNER_TEMP "native-nuget-packages"
New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
$downloadPath = Join-Path $downloadDirectory $packageFile

# Fetch the pinned package without NuGet's full version/dependency metadata queries.
$url = "$feedBase/flat2/$packageId/$packageVersion/$packageFile"
curl.exe --fail --location --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 --output $downloadPath $url
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download $PackageName $Version (curl exit code $LASTEXITCODE)."
}

# Retain the local feed for C# restore, but skip unused managed dependencies here.
nuget install $PackageName -Version $Version -Source $downloadDirectory -ExcludeVersion -NonInteractive -DependencyVersion Ignore
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install $PackageName $Version (NuGet exit code $LASTEXITCODE)."
}
