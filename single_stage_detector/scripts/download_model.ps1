
$DOWNLOAD_LINK = "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"
$SHA512 = "15c9f0bc1c8d64750712f86ffaded3b0bc6a87e77a395dcda3013d8af65b7ebf3ca1c24dd3aae60c0d83e510b4d27731f0526b6f9392c0a85ffc18e5fecd8a13"
$FILENAME = "resnext50_32x4d-7cdf4587.pth"
$FOLDER_PATH = "C:\Users\Usuario-PC\Desktop\ms\training\single_stage_detector\ssd\model\weights"

# Create folder if it doesn't exist
if (!(Test-Path -Path $FOLDER_PATH)) {
    New-Item -ItemType Directory -Path $FOLDER_PATH
}

# Download the file
$FilePath = Join-Path -Path $FOLDER_PATH -ChildPath $FILENAME
Invoke-WebRequest -Uri $DOWNLOAD_LINK -OutFile $FilePath -UseBasicParsing

# Compute SHA512 hash
$FileHash = Get-FileHash -Algorithm SHA512 -Path $FilePath

# Verify the hash
if ($FileHash.Hash -eq $SHA512) {
    Write-Host "SHA512 hash matches!"
} else {
    Write-Host "SHA512 hash does not match!"
}