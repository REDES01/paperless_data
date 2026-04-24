COMPOSE = docker compose -f docker/docker-compose.yaml

# ── CHI@TACC Object Storage ──────────────────
# Set these before running chi-* targets.
# Generate credentials in the Chameleon Jupyter environment (see provision notebook).
CHI_ENDPOINT ?= chi.tacc.chameleoncloud.org:7480
CHI_ACCESS_KEY ?= $(shell echo $$CHI_ACCESS_KEY)
CHI_SECRET_KEY ?= $(shell echo $$CHI_SECRET_KEY)
CHI_BUCKET ?= paperless-chi

.PHONY: up down restart status logs clean
.PHONY: up-db up-storage up-stream up-vector
.PHONY: check-pg check-minio check-redpanda check-qdrant
.PHONY: ingest ingest-iam augment-iam
.PHONY: chi-ingest chi-ingest-iam chi-augment-iam
.PHONY: chi-batch chi-batch-htr chi-batch-retrieval
.PHONY: generate generate-api generate-traffic generate-stop
.PHONY: demo-retrieval demo-htr
.PHONY: batch batch-htr batch-retrieval
.PHONY: validate chi-validate

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

# ── Ingestion to local MinIO (dev/testing) ────
#
# IAM handwriting dataset only. SQuAD was in the original design for
# retrieval model fine-tuning, but the retrieval path uses pretrained
# mpnet directly — no contrastive training ever consumed SQuAD data.
# Re-add an ingest-squad target here if a retriever fine-tune is built.

ingest: ingest-iam augment-iam
	@echo "Full ingestion pipeline complete (local MinIO)."

ingest-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python ingest_iam.py

augment-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python augment_iam.py

# ── Ingestion to CHI@TACC persistent storage ──

chi-ingest: chi-ingest-iam chi-augment-iam
	@echo "Full ingestion pipeline complete (CHI@TACC)."

chi-ingest-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm \
		-e MINIO_ENDPOINT=$(CHI_ENDPOINT) \
		-e MINIO_ACCESS_KEY=$(CHI_ACCESS_KEY) \
		-e MINIO_SECRET_KEY=$(CHI_SECRET_KEY) \
		-e MINIO_SECURE=true \
		-e MINIO_BUCKET=$(CHI_BUCKET) \
		paperless-ingest python ingest_iam.py

chi-augment-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm \
		-e MINIO_ENDPOINT=$(CHI_ENDPOINT) \
		-e MINIO_ACCESS_KEY=$(CHI_ACCESS_KEY) \
		-e MINIO_SECRET_KEY=$(CHI_SECRET_KEY) \
		-e MINIO_SECURE=true \
		-e MINIO_BUCKET=$(CHI_BUCKET) \
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
	docker build -t paperless-retrieval -f ./online_features/Dockerfile ./online_features
	docker run --rm --network docker_default \
		-e QDRANT_HOST=qdrant \
		-e QDRANT_PORT=6333 \
		paperless-retrieval

demo-htr:
	docker build -t paperless-htr -f ./online_features/Dockerfile.htr ./online_features
	docker run --rm paperless-htr

# ── Batch Pipeline (local MinIO) ──────────────

batch: batch-htr batch-retrieval
	@echo "Batch pipeline complete (local MinIO)."

batch-htr:
	docker build -t paperless-batch ./batch_pipeline
	docker run --rm --network docker_default \
		-e DB_DSN="host=postgres dbname=paperless user=user password=paperless_postgres" \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-batch python batch_htr.py

batch-retrieval:
	docker build -t paperless-batch ./batch_pipeline
	docker run --rm --network docker_default \
		-e DB_DSN="host=postgres dbname=paperless user=user password=paperless_postgres" \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-batch python batch_retrieval.py

# ── Batch Pipeline (CHI@TACC) ─────────────────

chi-batch: chi-batch-htr chi-batch-retrieval
	@echo "Batch pipeline complete (CHI@TACC)."

chi-batch-htr:
	docker build -t paperless-batch ./batch_pipeline
	docker run --rm --network docker_default \
		-e DB_DSN="host=postgres dbname=paperless user=user password=paperless_postgres" \
		-e MINIO_ENDPOINT=$(CHI_ENDPOINT) \
		-e MINIO_ACCESS_KEY=$(CHI_ACCESS_KEY) \
		-e MINIO_SECRET_KEY=$(CHI_SECRET_KEY) \
		-e MINIO_SECURE=true \
		-e MINIO_BUCKET=$(CHI_BUCKET) \
		paperless-batch python batch_htr.py

chi-batch-retrieval:
	docker build -t paperless-batch ./batch_pipeline
	docker run --rm --network docker_default \
		-e DB_DSN="host=postgres dbname=paperless user=user password=paperless_postgres" \
		-e MINIO_ENDPOINT=$(CHI_ENDPOINT) \
		-e MINIO_ACCESS_KEY=$(CHI_ACCESS_KEY) \
		-e MINIO_SECRET_KEY=$(CHI_SECRET_KEY) \
		-e MINIO_SECURE=true \
		-e MINIO_BUCKET=$(CHI_BUCKET) \
		paperless-batch python batch_retrieval.py

# ── Quality Validation ─────────────────

validate:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python validate_augmentation.py

chi-validate:
	docker build -t paperless-ingest ./ingestion
	docker run --rm \
		-e MINIO_ENDPOINT=$(CHI_ENDPOINT) \
		-e MINIO_ACCESS_KEY=$(CHI_ACCESS_KEY) \
		-e MINIO_SECRET_KEY=$(CHI_SECRET_KEY) \
		-e MINIO_SECURE=true \
		-e MINIO_BUCKET=$(CHI_BUCKET) \
		paperless-ingest python validate_augmentation.py

# ── Health checks ─────────────────────────────

check-pg:
	docker exec postgres pg_isready -U user -d paperless

check-minio:
	curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK"

check-redpanda:
	docker exec redpanda rpk cluster health

check-qdrant:
	curl -sf http://localhost:6333/healthz && echo "Qdrant OK"
