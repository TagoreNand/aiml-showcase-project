install:
	pip install -r requirements.txt

train:
	python -m app.ml.train_classifier

serve:
	uvicorn app.main:app --reload

dashboard:
	streamlit run frontend/dashboard.py

test:
	pytest -q
