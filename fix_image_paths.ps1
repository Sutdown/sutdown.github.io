$files = Get-ChildItem -Path "e:\project\SutdownBlog\content" -Recurse -Include "*.md"
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    if ($content -match '\(/img/') {
        $newContent = $content -replace '\(/img/', '(/images/'
        [System.IO.File]::WriteAllText($file.FullName, $newContent, [System.Text.Encoding]::UTF8)
        Write-Host "Updated: $($file.FullName)"
    }
}
Write-Host "Done!"
