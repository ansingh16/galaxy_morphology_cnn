.PHONY: install data train-small train-resnet eval serve test lint

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -e .
	pip install -r requirements.txt

data:
	python -m galaxy_cnn.download_data

train-small:
	python -m galaxy_cnn.train --model small --epochs 18

train-resnet:
	python -m galaxy_cnn.train --model resnet --epochs 12

train-resnet-ft:
	python -m galaxy_cnn.train --model resnet_ft --epochs 10 --lr 3e-4

eval:
	python -m galaxy_cnn.evaluate --model small
	python -m galaxy_cnn.evaluate --model resnet
	python -m galaxy_cnn.evaluate --model resnet_ft

serve:
	uvicorn galaxy_cnn.serve:app --port 8000

test:
	pytest -q

lint:
	ruff check src tests
