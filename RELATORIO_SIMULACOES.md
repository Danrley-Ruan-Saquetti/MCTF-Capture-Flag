# Relatório de Simulações — MCTF 3v3 (Heurístico vs. pyquaticus)

_Gerado em 2026-06-15 22:08 | 20 episódios por condição | seed=0_

## 1. Configuração experimental

- **Cenário:** Capture-the-Flag 3v3 (MCTF 2026), arena de 160 x 80 m, linha de _scrimmage_ em x = 80 m.
- **Dinâmica:** modelo `surveyor` (MOOS-IvP), velocidade máxima 3 m/s, raio de captura/tag 10 m, _cooldown_ de tag 60 s.
- **Término do episódio:** primeiro time a 20 capturas ou 600 s de jogo.
- **Time azul (proposto):** `Attacker` + `HybridAgent` + `Defender` heurísticos (modo `hard`).
- **Time vermelho (linha de base):** três `Heuristic_CTF_Agent` do pyquaticus.
- **Métricas:** capturas (placar), _grabs_ (posses de bandeira), _tags_ (marcações) e taxa de vitória do time azul.

> **Nota sobre `speedup`:** o `sim_speedup_factor` aplica a mesma ação por `speedup` subpassos de física (0,1 s cada). Logo ele não é apenas aceleração de relógio: a 10 cada decisão controla 1,0 s de movimento e a 30 controla 3,0 s, reduzindo a fidelidade do controle. Por isso ele é tratado como fator experimental.

## 2. Resultados

| Modo de ação | speedup | Oponente | Placar (A×V) | Caps/ep A | Caps/ep V | Grabs/ep A | Grabs/ep V | Tags/ep A | Tags/ep V | V–E–D   | Win% A | Vencedor |
| ------------ | ------- | -------- | ------------ | --------- | --------- | ---------- | ---------- | --------- | --------- | ------- | ------ | -------- |
| contínuo     | 10      | easy     | 82×2         | 4.10±0.77 | 0.10±0.30 | 4.85       | 1.70       | 11.55     | 1.55      | 20–0–0  | 100%   | AZUL     |
| contínuo     | 10      | medium   | 73×1         | 3.65±0.79 | 0.05±0.22 | 4.75       | 1.05       | 10.75     | 2.30      | 20–0–0  | 100%   | AZUL     |
| contínuo     | 10      | hard     | 43×48        | 2.15±1.19 | 2.40±1.11 | 4.25       | 4.45       | 7.10      | 5.15      | 7–4–9   | 35%    | VERMELHO |
| contínuo     | 30      | easy     | 79×0         | 3.95±0.74 | 0.00±0.00 | 4.60       | 1.25       | 11.85     | 1.30      | 20–0–0  | 100%   | AZUL     |
| contínuo     | 30      | medium   | 67×0         | 3.35±0.73 | 0.00±0.00 | 4.70       | 1.10       | 10.75     | 2.25      | 20–0–0  | 100%   | AZUL     |
| contínuo     | 30      | hard     | 71×47        | 3.55±1.12 | 2.35±0.73 | 5.05       | 3.90       | 7.60      | 3.90      | 14–5–1  | 70%    | AZUL     |
| discreto     | 10      | hard     | 10×0         | 0.50±0.50 | 0.00±0.00 | 1.05       | 0.10       | 5.35      | 0.80      | 10–10–0 | 50%    | AZUL     |
| discreto     | 30      | hard     | 8×0          | 0.40±0.49 | 0.00±0.00 | 1.00       | 0.05       | 5.35      | 0.95      | 8–12–0  | 40%    | AZUL     |
