TEST_PATH=./shortest_route/tests/
.PHONY: develop

test:
	py.test -m "not exclude and not integration" -x -v $(TEST_PATH)

lint:
	pycodestyle --max-line-length=120 --ignore=E252,E121,E123,E126,E226,E24,E704,E402,W503 shortest_route/

install:
	python3 setupy.py install

develop:
	python3 setup.py develop
