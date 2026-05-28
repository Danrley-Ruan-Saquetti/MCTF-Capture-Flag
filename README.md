# Campeonato MCTF - UDESC - Modelagem de Agentes

- Para configurar o ambiente, siga o [SETUP.md](./SETUP.md).
- Para executar a simulação, rode o script [test_heuristic_3v3.py](./test_heuristic_3v3.py).

### Comando

```bash
python test_heuristic_3v3.py [--render] [--episodes N] [--speedup X] [--opponent easy|medium|hard]
```

### Argumentos

- `--render`: ativa visualização `pygame`
- `--episodes N`: número de episódios (padrão `1`)
- `--speedup X`: fator de aceleração da simulação
- `--opponent easy|medium|hard`: dificuldade do time oponente (padrão `hard`)
