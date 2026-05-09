# Primeiros Passos com Pyquaticus para MCTF

Esta página fornece uma visão geral de como treinar agentes usando aprendizado por reforço profundo dentro do framework Pyquaticus para a competição Maritime Capture-the-Flag (MCTF).

## Treinando Agentes para Jogar MCTF

Um exemplo completo para treinar três agentes como uma equipe coordenada está incluído dentro do Pyquaticus:

```bash
rl_test/train_3v3.py
```

Este script utiliza **RLlib**. Se você é novo no RLlib, veja a documentação [aqui](https://docs.ray.io/en/latest/rllib/index.html).

- Certifique-se de que seu ambiente virtual esteja ativado.
- Execute: `python train_3v3.py`
- Os modelos são salvos em: `ray_tests/<checkpoint>/policies/<policy-name>`
- A frequência de salvamento é controlada em: `competition_train_example.py : line 112`

## Mapeamento de Policies para IDs de Agentes

Abaixo está um trecho de `rl_test/train_3v3.py` mostrando como os nomes das policies são mapeados para agentes individuais. O RLlib usa esse mapeamento para que cada agente do jogo receba a policy correta de aprendizado ou de oponente.

```py
#Esta é a função onde você irá mapear o agente para uma das policies definidas no dicionário abaixo na linha: 19
#Na função de mapeamento abaixo os três primeiros agentes 0,1,2 (time azul) são mapeados para policies que serão aprendidas
#O time vermelho é mapeado para a policy heurística onde os agentes sempre escolherão a ação de não fazer nada
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
  if agent_id == 0 or agent_id == 'agent-0':
    return "agent-0-policy"
  if agent_id == 1 or agent_id == 'agent-1':
    return "agent-1-policy"
  elif agent_id == 2 or agent_id == 'agent-2':
    # altere isto para agent-1-policy para treinar ambos os agentes ao mesmo tempo
    return "agent-2-policy"
  elif agent_id == 3 or agent_id == 'agent-3':
    return "noop-policy-3"
  elif agent_id == 4 or agent_id == 'agent-4':
    return "noop-policy-4"
  else:
    return "noop-policy-5"

#A função de mapeamento de policies estabelece quais policies estarão aprendendo e quais serão baseadas em heurística (usadas no treinamento)
#Se você estiver usando uma policy heurística durante o treinamento, certifique-se de passar a policy associada ao ID correto do agente
#Por exemplo, queremos treinar contra agentes que não fazem nada a cada turno; para isso precisamos criar noop-policy-3, noop-policy-4 e noop-policy-5
#**Nota: Para policies heurísticas usadas no treinamento de agentes RL, adicione {"no_checkpoint": True} às policies

policies = {'agent-0-policy':(None, obs_space, act_space, {}),
      'agent-1-policy':(None, obs_space, act_space, {}),
      'agent-2-policy':(None, obs_space, act_space, {}),
      'noop-policy-3':(AttackGen(3, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True}),
      'noop-policy-4':(AttackGen(4, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True}),
      'noop-policy-5':(AttackGen(5, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True}),
      'easy-defend-policy': (DefendGen(4, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True}),
      'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {"no_checkpoint": True})}
```

## Algoritmo de Treinamento: Rollout Workers & GPUs

A seguinte configuração PPO (de train_3v3.py) determina os recursos computacionais e associa policies aos agentes durante o treinamento.

```py
#Cria uma configuração PPO do rllib; aqui estão alguns dos algoritmos já implementados pelo rllib: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
#Nesta configuração temos 1 ambiente rodando em 1 CPU; esses valores devem ser atualizados para refletir os recursos computacionais disponíveis no seu sistema
#Não utilizando o Alpha Rllib (api_stack False)

ppo_config = PPOConfig()
  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
  .environment(env='pyquaticus')
  .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)

#Se o seu sistema permitir, alterar o número de rollouts pode reduzir significativamente os tempos de treinamento (num_rollout_workers=15)
ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"],)
algo = ppo_config.build_algo()
```

- Modifique a linha 3 para ajustar os recursos de CPU/GPU.
- Modifique a linha 7 para alterar os nomes das policies treinadas.

## Design da Função de Recompensa

O reward shaping é crucial no treinamento multiagente baseado em RL. O Pyquaticus inclui vários exemplos de funções de recompensa em:

`pyquaticus/envs/utils/rewards.py`

```txt
#Recompensas Configuráveis
  # -- NOTA --
  # Todos os headings estão no formato náutico
  # 0
  # |
  # 270 -- . -- 90
  # |
  # 180
  #
  # Isso pode ser convertido para o formato padrão de heading anti-horário
  # usando a função heading_angle_conversion(deg) encontrada em utils.py
  #
  #
  ## Cada função de recompensa customizada deve possuir os seguintes argumentos
  Args:
    agent_id (int): ID do agente para o qual estamos computando a recompensa
    team (Team): time do agente para o qual estamos computando a recompensa
    agents (list): lista de IDs dos agentes (usada para mapear agent_id para índices de agentes e vice-versa)
    agent_inds_of_team (dict): mapeamento do time para os índices dos agentes daquele time
    state (dict):
      'agent_position' (array): lista das posições dos agentes (indexadas na ordem da lista agents)

        Ex. Uso: Obter a posição atual do agente
        agent_id = 'agent_1'
        position = state['agent_position'][agents.index(agent_id)]

      'prev_agent_position' (array): lista das posições dos agentes no timestep anterior (indexadas na ordem da lista agents)

        Ex. Uso: Obter a posição anterior do agente
        agent_id = 'agent_1'
        prev_position = state['prev_agent_position'][agents.index(agent_id)]

      'agent_speed' (array): lista das velocidades dos agentes (indexadas na ordem da lista agents)

        Ex. Uso: Obter a velocidade do agente
        agent_id = 'agent_1'
        speed = state

      'agent_heading' (array): lista dos headings dos agentes (indexadas na ordem da lista agents)

        Ex. Uso: Obter o heading do agente
        agent_id = 'agent_1'
        heading = state['agent_heading'][agents.index(agent_id)]

      'agent_on_sides' (array): lista de booleanos (indexadas na ordem da lista agents) onde True significa que o agente está no seu próprio lado, e False significa que não está

        Ex. Uso: Verificar se o agente está no seu próprio lado
        agent_id = 'agent_1'
        on_own_side = state['agent_on_sides'][agents.index(agent_id)]

      'agent_oob' (array): lista de booleanos (indexadas na ordem da lista agents) onde True significa que o agente está fora dos limites (OOB), e False significa que não está

        Ex. Uso: Verificar se o agente está fora dos limites
        agent_id = 'agent_1'
        num_oob = state['agent_oob'][agents.index(agent_id)]

      'agent_has_flag' (array): lista de booleanos (indexadas na ordem da lista agents) onde True significa que o agente possui uma bandeira, e False significa que não possui

        Ex. Uso: Verificar se o agente possui uma bandeira
        agent_id = 'agent_1'
        has_flag = state['agent_has_flag'][agents.index(agent_id)]

      'agent_is_tagged' (array): lista de booleanos (indexadas na ordem da lista agents) onde True significa que o agente foi marcado/tagueado, e False significa que não foi

        Ex. Uso: Verificar se o agente foi marcado
        agent_id = 'agent_1'
        is_tagged = state['agent_is_tagged'][agents.index(agent_id)]

      'agent_made_tag' (array): lista (indexada na ordem da lista agents) onde o valor em uma entrada é o índice de um agente diferente

        que foi marcado pelo agente correspondente naquele timestep atual, caso contrário None
        Ex. Uso: Verificar se o agente marcou outro agente
        agent_id = 'agent_1'
        tagged_opponent_idx = state['agent_made_tag'][agents.index(agent_id)]

      'agent_tagging_cooldown' (array): cooldown atual de marcação dos agentes (indexado na ordem da lista agents)

        Nota: o agente pode marcar quando esse valor é igual a tagging_cooldown
        Ex. Uso: Obter o cooldown atual de marcação do agente
        agent_id = 'agent_1'
        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]

      'dist_bearing_to_obstacles' (dict): Para cada agente na partida, lista as distâncias e direções para todos os obstáculos na ordem da lista de obstáculos

      'flag_home' (array): lista das bases das bandeiras (indexadas pelo número do time)

      'flag_position' (array): lista das posições das bandeiras (indexadas pelo número do time)

      'flag_taken' (array): lista de booleanos (indexadas pelo número do time) onde True significa que a bandeira do time foi capturada (pegada por um oponente), e False significa que não foi

      'team_has_flag' (array): lista de booleanos (indexadas pelo número do time) onde True significa que um agente do time possui uma bandeira, e False significa que nenhum agente possui uma bandeira

      'captures' (array): lista do total de capturas realizadas por cada time (indexadas pelo número do time)

      'tags' (array): lista do total de tags realizadas por cada time (indexadas pelo número do time)

      'grabs' (array): lista do total de capturas de bandeira realizadas por cada time (indexadas pelo número do time)

      'agent_collisions' (array): lista do total de colisões para cada agente (indexadas na ordem da lista agents)

      'agent_dynamics' (array): lista de dicionários contendo informações dinâmicas específicas dos agentes (atributo state de uma classe de dinâmica - veja dynamics.py)

      ######################################################################################
      ##### As seguintes chaves existirão no dicionário state se lidar_obs for True #####
        'lidar_labels' (dict):

        'lidar_labels' (dict):

        'lidar_labels' (dict):
      ######################################################################################

      'obs_hist_buffer' (dict): Buffer de histórico de observações onde as chaves são os agent_id's e os valores são as observações dos agentes

      'global_state_hist_buffer' (array): Buffer de histórico do estado global

    prev_state (dict): Contém as informações do estado do passo anterior

    env_size (array): dimensões do campo [horizontal, vertical]

    agent_radii (array): lista dos raios dos agentes (indexadas na ordem da lista agents)

    catch_radius (float): raio para tag e captura de bandeira

    scrimmage_coords (array): pontos finais [x,y] da linha de scrimmage

    max_speeds (list): lista das velocidades máximas dos agentes (indexadas na ordem da lista agents)

    tagging_cooldown (float): tempo de cooldown para marcação
```

Abaixo está um exemplo de uma função de recompensa esparsa que utiliza tanto state quanto prev_state para determinar transições e atribuir recompensas:

```py
def caps_and_grabs(
  agent_id: str,
  team: Team,
  agents: list,
  agent_inds_of_team: dict,
  state: dict,
  prev_state: dict,
  env_size: np.ndarray,
  agent_radius: np.ndarray,
  catch_radius: float,
  scrimmage_coords: np.ndarray,
  max_speeds: list,
  tagging_cooldown: float
):
  reward = 0.0
  prev_num_oob = state['agent_oob'][agents.index(agent_id)]
  num_oob = state['agent_oob'][agents.index(agent_id)]

  if num_oob > prev_num_oob:
    reward += -1.0

  for t in state['grabs']:
    prev_num_grabs = state['grabs'][t]
    num_grabs = state['grabs'][t]

    if num_grabs > prev_num_grabs:
      reward += 0.25 if t == team else -0.25

    prev_num_caps = state['captures'][t]
    num_caps = state['captures'][t]

    if num_caps > prev_num_caps:
      reward += 1.0 if t == team else -1.0

  return reward
```
