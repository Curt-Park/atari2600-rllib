setup:
	python3 -m pip install -r requirements.txt
	AutoROM -y

format:
	brunette . --config=setup.cfg
	isort .

lint:
	pytest . --pylint --flake8 --mypy

docker-run:
	docker run -it --gpus all -v $(shell pwd):/home/ray/atari2600 rayproject/ray-ml:1.9.0 /bin/bash
