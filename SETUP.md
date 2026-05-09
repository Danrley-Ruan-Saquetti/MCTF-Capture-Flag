## Configuração e Instalação do Software MCTF

O MCTF utiliza o ambiente de simulação de jogos Pyquaticus. O Pyquaticus é uma biblioteca open-source baseada em Python, desenvolvida pelo MIT Lincoln Laboratory, que permite treinamento multiagente via aprendizado por reforço (RL) utilizando um ambiente PettingZoo/Gymnasium. Ele oferece suporte à biblioteca de deep RL RLlib e pode integrar-se com outras bibliotecas, como Stable-Baselines ou implementações personalizadas de RL. O Pyquaticus também oferece suporte a agentes baseados em heurística e controle manual dos agentes via teclado.

## Etapas para Baixar e Instalar o Pyquaticus

1. Baixe o Pyquaticus a partir [do repositório no GitHub](https://github.com/mit-ll-trusted-autonomy/pyquaticus/tree/main).
2. `git checkout mctf2026` (OBS: `mctf2026` é a branch que será utilizada para a competição de 2026, e não a `main`)
3. Siga as instruções no arquivo `README.md`.
4. Configure um ambiente virtual Python utilizando um dos métodos abaixo.
5. Treine seus primeiros agentes de RL utilizando o guia [Getting Started](/TRAIN_AGENT.md).

## Configurando um Ambiente Virtual Python

### Opção 1: Utilizando Miniconda (Recomendado)

1. Instale o Miniconda a partir [deste link](https://docs.anaconda.com/miniconda/miniconda-install/).
2. Adicione o Miniconda ao System PATH:
    - C:\Users\<user>\miniconda3
    - C:\Users\<user>\miniconda3\Scripts
    - C:\Users\<user>\miniconda3\Library\bin
3. Navegue até o repositório clonado do Pyquaticus.
4. Execute:
   `./setup-conda-env.sh light`
   `./setup-conda-env.sh full` (recomendado, inclui RLlib)

### Opção 2: Utilizando Ambiente Virtual Python

1. Instale o Python 3.10.
2. Crie um ambiente virtual:
   `python3.10 -m venv <envname>`
3. Instale o Pyquaticus:
   `pip install -e .[torch,ray]` (completo)
   ou `pip install -e .` (leve)

## Configuração no Windows

1. Instale o Miniconda a partir [deste link](https://docs.anaconda.com/miniconda/miniconda-install/).
2. Adicione o Miniconda ao System PATH:
    - C:\Users\<user>\miniconda3
    - C:\Users\<user>\miniconda3\Scripts
    - C:\Users\<user>\miniconda3\Library\bin
3. Após a instalação, o python instalado via Miniconda deve ser da versão latest do Python, por isso, execute o seguinte comando para criar um ambiente virtual com a versão 3.10 do Python:
   `conda create -n py310 python=3.10`
4. Rode o comando `conda init` e reinicie o terminal
5. E então confira a versão do python com `python --version` (deve ser 3.10.x)
6. Clone o repositório GitHub do Pyquaticus.
7. Navegue até o repositório.
8. `git checkout mctf2026` (OBS: `mctf2026` é a branch que será utilizada para a competição de 2026, e não a `main`)
9. Remova a linha 56 (`pymoos==2022.1`) do arquivo `pyproject.toml`.
10. Execute:
    `./setup-conda-env.sh light` (WSL)
    ou
    `./setup-conda-env.sh full` (inclui RLlib)
11. Ou instale manualmente:
    `pip install -e .[torch,ray]`
    ou
    `pip install -e .`
