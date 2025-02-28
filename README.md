<div align="left">
    <img src="./svg/pytorch-lightning.svg" width="40%" align="left" style="margin-right: 15px"/>
    <div style="display: inline-block;">
        <h2 style="display: inline-block; vertical-align: middle; margin-top: 0;">LIGHTNING-PRACTICE</h2>
        <p>
	<em>Эталонное решение практики</em>
</p>
        <p>
	<img src="https://img.shields.io/github/last-commit/cosheimil/Lightning-Practice?style=flat-square&logo=git&logoColor=white&color=A931EC" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/cosheimil/Lightning-Practice?style=flat-square&color=A931EC" alt="repo-top-language">
</p>
        <p>Built with the tools and technologies:</p>
        <p>
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/Lightning-792EE5.svg?style=flat-square&logo=Lightning&logoColor=white" alt="Lightning">
</p>
    </div>
</div>
<br clear="left"/>

## Содержание

- [Обзор](#-обзор)
- [Структура проекта](#-структура-проекта)
- [Запуск проекта](#-запуск-проекта)
  - [Необходимые инструменты](#-необходимые-инструменты)
  - [Установка зависимостей](#-установка-зависимостей)
  - [Тестирование](#-тестирование)

## Обзор

Данный проект является эталонным решением практики на [Stepik]("https://stepik.org/lesson/1466624/step/15?unit=1486235").
Если вы нашли какой-то недочет в проекте - нашите в группу курса 😊


## Структура проекта

```sh
└── Lightning-Practice/
    ├── data
    │   ├── sign_mnist_test.csv
    │   └── sign_mnist_train.csv
    ├── pyproject.toml
    ├── src
    │   ├── convnet.py
    │   ├── datamodule.py
    │   ├── dataset.py
    │   ├── main.py
    │   └── trainer.py
    └── uv.lock
```

## Запуск проекта

### Необходимые инструменты

Чтобы запустить этот проект вам понадобится установленный `Python`

### Установка зависимостей

Чтобы установить зависимости воспользуйтесь инструментом [`uv`]("https://github.com/astral-sh/uv")

1. Склонируйте репозиторий:
```sh
❯ git clone https://github.com/cosheimil/Lightning-Practice
```

2. Перейдите в директорию:
```sh
❯ cd Lightning-Practice
```

3. Создайте `venv`:
```sh
❯ uv sync
```

### Тестирование
Чтобы запустить проект:
```sh
❯ python src/main.py
```
