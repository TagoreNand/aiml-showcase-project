.PHONY: install install-optional train eval serve dashboard test seed lint \
        compose-up compose-full k8s-apply k8s-delete

install:
	pip install -r requirements.txt

install-optional:
	pip install -r requirements-optional.txt
	python -m spacy download en_core_web_sm || true

train:
	python -m app.ml.train_classifier

eval:
	python -m app.ml.evaluate_classifier

serve:
	uvicorn app.main:app --reload

dashboard:
	streamlit run frontend/dashboard.py

test:
	pytest -q

seed:
	python -m scripts.seed_sample_data

lint:
	python -m pyflakes app || true

# --- containers / orchestration ---
compose-up:            ## light, in-memory API only
	docker compose up --build api

compose-full:          ## Postgres+pgvector, MLflow, Prometheus, Grafana
	docker compose --profile full up --build

k8s-apply:
	kubectl apply -k deploy/k8s

k8s-delete:
	kubectl delete -k deploy/k8s
