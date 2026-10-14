# oagbench CLI and dashboard. Phase 3 review: pin base images by digest and scan with Trivy.
FROM ghcr.io/astral-sh/uv:0.11 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
      --extra anthropic --extra openai --extra services --extra dashboard
COPY src ./src
# Non-editable, so the venv is self-contained when copied into the runtime image.
RUN uv sync --frozen --no-dev --no-editable \
      --extra anthropic --extra openai --extra services --extra dashboard

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 DATA_DIR=/work/data RESULTS_DIR=/work/results
WORKDIR /work
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1
ENTRYPOINT ["oagbench"]
CMD ["status"]
