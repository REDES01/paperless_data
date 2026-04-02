COMPOSE = docker compose -f docker/docker-compose.yaml
CACHE_DIR = $(HOME)/paperless_cache

.PHONY: up down restart status logs clean
.PHONY: up-db up-storage up-stream up-vector
.PHONY: check-pg check-minio check-redpanda check-qdrant
.PHONY: ingest ingest-iam ingest-squad augment-iam download-cache
.PHONY: generate generate-api generate-traffic generate-stop
.PHONY: demo-retrieval demo-htr

# ── Lifecycle ─────────────────────────────────

up:
	$(COMPOSE) up -d
	@echo ""
	@echo "  MinIO Console   ->  http://localhost:9001  (admin / paperless_minio)"
	@echo "  Adminer (PG)    ->  http://localhost:5050  (user / paperless_postgres)"
	@echo "  Redpanda UI     ->  http://localhost:8090"
	@echo "  Qdrant          ->  http://localhost:6333/dashboard"

down:
	$(COMPOSE) down

restart: down up

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

clean:
	$(COMPOSE) down -v

# ── Bring up individual layers ────────────────

up-db:
	$(COMPOSE) up -d postgres adminer

up-storage:
	$(COMPOSE) up -d minio minio-init

up-stream:
	$(COMPOSE) up -d redpanda redpanda-init redpanda-console

up-vector:
	$(COMPOSE) up -d qdrant

# ── Ingestion Pipeline ────────────────────────

download-cache:
	@mkdir -p $(CACHE_DIR)
	@test -f $(CACHE_DIR)/train-v2.0.json || \
		(echo "Downloading SQuAD train..." && \
		 wget -q -O $(CACHE_DIR)/train-v2.0.json \
		 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
	@test -f $(CACHE_DIR)/dev-v2.0.json || \
		(echo "Downloading SQuAD dev..." && \
		 wget -q -O $(CACHE_DIR)/dev-v2.0.json \
		 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
	@echo "Cache ready: $(CACHE_DIR)"

ingest: ingest-iam ingest-squad augment-iam
	@echo "Full ingestion pipeline complete."

ingest-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python ingest_iam.py

ingest-squad: download-cache
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		-e CACHE_DIR=/cache \
		-v $(CACHE_DIR):/cache:ro \
		paperless-ingest python ingest_squad.py

augment-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python augment_iam.py

# ── Data Generator ────────────────────────────

generate: generate-api generate-traffic

generate-api:
	docker build -t paperless-datagen ./data_generator
	docker run -d --name datagen-api --network docker_default \
		-e DB_DSN="host=postgres dbname=paperless user=user password=paperless_postgres" \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		-e KAFKA_BROKER=redpanda:9092 \
		-p 8000:8000 \
		paperless-datagen
	@echo "Stub API running on http://localhost:8000"
	@sleep 3

generate-traffic:
	docker run --rm --name datagen-traffic --network docker_default \
		paperless-datagen python generator.py \
		--api-url http://datagen-api:8000 \
		--rate 2.0 \
		--duration 300

generate-stop:
	-docker stop datagen-api
	-docker rm datagen-api

# ── Online Feature Computation ────────────────

demo-retrieval:
	docker build -t paperless-features ./online_features
	docker run --rm --network docker_default \
		-e QDRANT_HOST=qdrant \
		-e QDRANT_PORT=6333 \
		paperless-features python retrieval_features.py

demo-htr:
	docker build -t paperless-features ./online_features
	docker run --rm --network docker_default \
		paperless-features python htr_features.py

# ── Health checks ─────────────────────────────

check-pg:
	docker exec postgres pg_isready -U user -d paperless

check-minio:
	curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK"

check-redpanda:
	docker exec redpanda rpk cluster health

check-qdrant:
	curl -sf http://localhost:6333/healthz && echo "Qdrant OK"
