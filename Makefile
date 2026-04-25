venv:
	bash scripts/create_venv.sh .uv

install:
	. .uv/bin/activate && python -m pip install -r requirements.txt

clean:
	rm -rf .venv __pycache__ .pytest_cache
