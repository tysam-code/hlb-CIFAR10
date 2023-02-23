install: 
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black main.py

lint:
	pylint -E --ignore-comments y --output errors.log main.py

codestyle:
	pycodestyle --show-source --show-pep8 main.py