# Multi-Drone Cooperative Search — Reinforcement Learning Project 2025/26

Progetto finale per il corso di **Reinforcement Learning** (Università di Bologna, A.A. 2025/26).

Un numero configurabile di droni deve trovare un bersaglio nascosto su una griglia con ostacoli casuali, nel minor tempo possibile, usando **Independent DQN con parameter sharing**.

---

## Caratteristiche principali

- **Ambiente stocastico**: dimensione della griglia, densità degli ostacoli e posizione del bersaglio cambiano a ogni episodio
- **Osservabilità parziale**: ogni drone vede solo le celle nel proprio raggio visivo e mantiene una mappa interna accumulata
- **Comunicazione di prossimità**: quando due o più droni sono entro `comm_range`, le loro mappe vengono fuse automaticamente
- **Algoritmo**: Independent DQN con parameter sharing (baseline)
- **GUI**: rendering in tempo reale con pygame

## Struttura del progetto

```
project/
├── env/               # Ambiente Gymnasium custom
│   ├── grid_env.py    # DroneSearchEnv
│   └── utils.py       # Generazione griglia e BFS
├── agents/            # Agente DQN
│   └── dqn_agent.py   # QNetwork, ReplayBuffer, DQNAgent
├── training/
│   └── train.py       # Loop di training e valutazione
├── gui/
│   └── renderer.py    # Rendering pygame
├── configs/
│   └── default.yaml   # Parametri configurabili
├── main.py            # Entry point
└── requirements.txt
```

## Installazione

```bash
# Opzione A — conda (consigliata: include PyTorch + CUDA 11.8)
conda env create -f environment.yaml
conda activate rl-drone

# Opzione B — pip
pip install -r requirements.txt
```

## Utilizzo

```bash
# Training
python main.py --mode train

# Valutazione con GUI (richiede checkpoint)
python main.py --mode eval --checkpoint checkpoints/best.pt

# Modalità gioco (drone 0 controllato da tastiera)
python main.py --mode play
```

## Parametri configurabili

Tutti i parametri si trovano in `configs/default.yaml`:

| Parametro | Default | Descrizione |
|---|---|---|
| `grid_min_size` / `grid_max_size` | 8 / 15 | Intervallo dimensione griglia |
| `obstacle_density` | 0.2 | Percentuale ostacoli |
| `n_agents` | 3 | Numero di droni |
| `vision_radius` | 2 | Raggio visivo |
| `comm_range` | 3 | Distanza massima comunicazione |
| `comm_metric` | manhattan | Metrica distanza (manhattan/euclidean) |
| `max_steps` | 200 | Passi massimi per episodio |
| `seed` | 42 | Seed globale |

## Metriche di training

Le curve di apprendimento sono disponibili su TensorBoard:

```bash
tensorboard --logdir runs/
```
