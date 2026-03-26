FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY sqa_middleware/ ./sqa_middleware/
COPY benchmarks/     ./benchmarks/

# Results volume mount point
RUN mkdir -p /results

CMD ["python", "-m", "benchmarks.ab_test_runner", "--output", "/results/benchmark_results.json"]
