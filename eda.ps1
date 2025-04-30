# PowerShell script to build and run a Docker container for YOLOv8 FastAPI

$ImageName = "eda:latest"
$Port = 8000
$LocalDataPath = Join-Path $PWD "data"
$ContainerDataPath = "/app/data"

# Step 1: Build Docker image
Write-Host "🔧 Building Docker image '$ImageName'..."
docker build -t $ImageName ./data_processing/EDA

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker build failed!"
    exit 1
}

# Step 2: Run Docker container
Write-Host "🚀 Running Docker container with port $Port exposed and volume mounted..."
docker run -it `
    -p "${Port}:${Port}" `
    -v "${LocalDataPath}:${ContainerDataPath}" `
    $ImageName `
    bash
