install:
	poetry install --no-dev

install-dev:
	poetry install

fmt:
	poetry run black adversarial_attack
	poetry run black tests

lint:
	poetry run black --check adversarial_attack
	poetry run flake8 adversarial_attack
	poetry run mypy adversarial_attack
	poetry run black --check tests
	poetry run flake8 tests
	poetry run mypy tests

clean:
	rm -rf .mypy_cache adversarial_attack/*.egg-info 
	find . -type d -name '__pycache__' -exec rm -r {} +
	
