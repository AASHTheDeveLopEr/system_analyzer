<div align="center">

# System Analyzer

### A local tool for reading system status

Inspects the machine it runs on and reports the main resources in one place: processor, memory, disk, and the current runtime.

<br>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Local](https://img.shields.io/badge/Runs-Locally-2C3E50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

</div>

---

# About

**System Analyzer** is a small local program for checking the computer it is started on.

It is meant for a quick status read before a heavier task: how much memory is free, how the processor looks, and whether disk space is tight. It does not phone home and it does not need an account.

> A local system status tool for CPU, memory, disk, and runtime checks.

---

# Features

- Reads the host it is running on
- Reports processor and memory
- Reports disk space
- Prints a short status instead of a dashboard login
- Stays on the machine

---

# Project structure

```text
system_analyzer/
├── main file
└── README.md
```

Replace main file with the real entry script after clone.

---

Requirements

· Python 3.8+
· The packages listed in the project, if a requirements file is present

---

Run

```bash
git clone https://github.com/AASHTheDeveLopEr/system_analyzer.git
cd system_analyzer
python main.py
```

If the entry file has another name, run that file instead.

---

Notes

· Output depends on the operating system.
· Some readings need permission to see process or disk details.
· Do not commit tokens, passwords, or machine-specific secrets.

---

License

Released under the MIT License when a LICENSE file is added to the repository.

```
