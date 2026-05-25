# Docker Guide — Used Car Price API

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- The trained CatBoost model (`models/catboost_final.cbm`) and API artifacts (`models/api_artifacts/`) in place

## Building and Running

Start the application:

```bash
docker compose up --build
```

The API will be available at:

- Root: http://localhost:8000/
- Swagger docs: http://localhost:8000/api/v1/docs
- Health check: http://localhost:8000/health
- Readiness check: http://localhost:8000/ready
- Predict endpoint: http://localhost:8000/api/v1/predict
- Metrics: http://localhost:8000/metrics

To run in the background (detached mode):

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

## Configuration

The container reads configuration from environment variables. To customize, create a `.env` file in the project root (see `.env.example` for available options). The `.env` file is automatically loaded by Docker Compose.

If no `.env` file is present, the application uses sensible defaults for local development.

## Rebuilding

After changing application code or dependencies:

```bash
docker compose up --build
```

For a fully clean rebuild (no cached layers):

```bash
docker compose build --no-cache
docker compose up
```

## Viewing Logs

```bash
docker compose logs         # all logs
docker compose logs -f      # follow logs in real time
```

## Deploying to the Cloud

The CI pipeline automatically builds and publishes the image to GitHub Container Registry on every push to `main`. You can pull it directly:

```bash
docker pull ghcr.io/erin-weiss/used-car-price-api:latest
```

To build and push manually (e.g., for a different registry or architecture):

```bash
# Build the image
docker build -t used-car-price-api .

# If your cloud uses a different CPU architecture than your development machine
# (e.g., Mac M1/M2 deploying to amd64):
docker build --platform=linux/amd64 -t used-car-price-api .

# Tag for your registry
docker tag used-car-price-api ghcr.io/erin-weiss/used-car-price-api:latest

# Push (requires authentication — CI handles this automatically)
docker push ghcr.io/erin-weiss/used-car-price-api:latest
```

## Health Checks

The container includes a built-in health check that pings `/health` every 30 seconds. Docker Desktop and Kubernetes will automatically monitor container health using this.

## References

- [Docker's Python guide](https://docs.docker.com/language/python/)
- [Docker Compose reference](https://docs.docker.com/go/compose-spec-reference/)