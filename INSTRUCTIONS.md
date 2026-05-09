# Regras e Pontuação do Jogo MCTF

O Maritime Capture-the-Flag (MCTF) é uma competição em equipes na qual veículos marinhos autônomos competem para capturar a bandeira do oponente enquanto defendem a sua própria.
As partidas são realizadas dentro de um ambiente marítimo delimitado e possuem duração fixa de **10 minutos**.

## Visão Geral do Jogo

O campo de jogo é uma área retangular medindo **160 m × 80 m**, dividida em duas metades iguais de **80 m × 80 m**. Duas equipes, **Vermelha** e **Azul**, competem em cada partida. Cada equipe é composta por **três agentes autônomos**.

Cada equipe possui duas **bases iniciais** em formato de quarto de círculo localizadas nos dois cantos mais distantes da linha central. Os agentes devem estar dentro de um diâmetro de **10 m** dos cantos para capturar/desmarcar com sucesso.

Os agentes podem se mover livremente dentro do campo de jogo. Qualquer agente que saia dos limites do campo é automaticamente retornado para a base inicial mais próxima.

## Eventos do Jogo

### Tag

Um tag ocorre quando um agente chega a menos de **10 metros** de um agente adversário _dentro da metade do campo pertencente à sua própria equipe_. Ambos os agentes devem estar na mesma metade para que o tag seja contabilizado.

Quando marcado, um agente deve retornar a uma de suas bases iniciais para se desmarcar (o jogo automaticamente conduzirá o agente de volta; o controle é devolvido quando o agente não estiver mais marcado). Se o agente estiver carregando a bandeira do oponente, a bandeira é imediatamente retornada para a base inicial adversária.

### Captura da Bandeira (Flag Grab)

Um agente pode pegar a bandeira do oponente ao chegar a menos de **10 metros** da bandeira da equipe adversária (círculos Azul e Vermelho), estando em estado não marcado e desde que a bandeira esteja presente.

### Captura de Bandeira (Flag Capture)

Uma captura de bandeira ocorre quando um agente retorna com sucesso a uma de suas bases iniciais carregando a bandeira adversária e sem estar marcado.

### Auto-Tag

Se um agente sair dos limites do campo de jogo, um tag automático é aplicado. O agente é imediatamente retornado para a base inicial mais próxima.

### Desmarcação

Agentes marcados — seja por um oponente ou via auto-tag — devem retornar à sua base inicial para se desmarcarem. Agentes marcados não podem marcar adversários nem interagir com bandeiras.

Quando um agente é marcado, o controlador do jogo substitui temporariamente os controles do agente e o navega de volta para sua base inicial.

## Regras de Powerplay

Cada equipe recebe um **Powerplay** por partida. Um Powerplay dura **2 minutos**, e os períodos de Powerplay das equipes adversárias **não se sobrepõem**.

- Durante o Powerplay de uma equipe, **um agente da equipe adversária é desabilitado**.
- O agente desabilitado não pode se mover, marcar ou interagir com bandeiras.
- O agente desabilitado é automaticamente reabilitado quando o Powerplay termina.
- Os Powerplays são aplicados pelo controlador do jogo.

## Duração da Partida & Pontuação

Cada partida dura um total de **10 minutos**.

### Pontuação Principal

```
Pontuação Principal = Capturas de Bandeira da Equipe - Capturas de Bandeira do Oponente
```

### Critério de Desempate

1. Capturas de Bandeira (maior quantidade)
2. Colisões (menor quantidade)

### Definições de Pontuação

**Captura de Bandeira:** Retornar a bandeira adversária para a base inicial  
**Flag Grab:** Pegar com sucesso a bandeira adversária  
**Tag:** Marcar um agente adversário dentro da distância de marcação  
**Colisão:** Qualquer ocorrência em que um agente fique a menos de **4 metros** de outro agente (companheiro de equipe ou adversário)

## Rodadas da Competição

### Simulação

Cada agente enviado é avaliado por meio de um torneio round-robin contra as 10 melhores equipes atuais. Os rankings são atualizados no placar conforme os resultados. Se uma submissão alcançar o top 10, a avaliação é repetida para recalcular as posições entre as 10 melhores equipes e garantir consistência no ranking.

### Hardware

As 10 melhores equipes da rodada de simulação competirão então em um torneio round-robin utilizando veículos de superfície não tripulados (USV) na United States Military Academy (USMA).

## Resultados

Os resultados são registrados e exibidos na página inicial do MCTF 2026 para as competições de simulação e hardware.
As 10 equipes com melhor desempenho serão convidadas a participar de um artigo acadêmico destacando algoritmos/abordagens e os resultados da competição.

Não há prêmios monetários ou não monetários por participação ou colocação.

## Desclassificação & Termos Legais

Os organizadores da competição reservam-se o direito de desclassificar qualquer submissão que viole as regras da competição ou os requisitos de elegibilidade. Os participantes não podem criar múltiplas contas.

A participação não obriga os organizadores ou instituições afiliadas a adquirir quaisquer soluções submetidas. Todos os participantes concordam em cumprir as leis e regulamentos aplicáveis.

### Requisitos de Elegibilidade

Você é elegível para se registrar e competir neste desafio somente se atender a todos os seguintes requisitos:

- Você concorda com as regras descritas neste desafio. Qualquer participante que envie materiais para consideração concorda implicitamente com as regras descritas neste desafio.
- Você possui 18 anos de idade ou mais.
- Regulamentos incluídos na página de sanções do OFAC (Office of Foreign Assets Control). Residentes desses países/regiões que não sejam funcionários governamentais podem participar.
- Você não está atualmente listado na U.S. Statutory Debarment List do DDTC (Directorate of Defense Trade Controls).
- Caso você seja uma entidade federal ou funcionário federal, poderá participar da competição MCTF.
- Os participantes devem estar registrados em equipes mutuamente exclusivas. Cada equipe pode enviar apenas uma entrada final (salvo aprovação de um organizador da competição).
- Os organizadores da competição não se responsabilizam por inscrições que não sejam recebidas por qualquer motivo, ou por inscrições recebidas que resultem em erro durante a execução.

### Renúncia de Reivindicações

O participante concorda em liberar e renunciar permanentemente a toda e qualquer reivindicação, ajustes equitativos, ações, processos, dívidas, recursos e quaisquer outras obrigações de qualquer natureza, passadas ou presentes, conhecidas ou desconhecidas, que possam surgir de, estar relacionadas a, ou estar conectadas direta ou indiretamente a este desafio ou à submissão do participante.

### Conformidade com as Leis

O participante concorda em seguir e cumprir todas as leis, regulamentos e políticas federais, estaduais e locais aplicáveis.

### Lei Aplicável

Esta competição está sujeita a todas as leis e regulamentos federais aplicáveis. TODAS AS REIVINDICAÇÕES DECORRENTES OU RELACIONADAS A ESTES TERMOS SERÃO REGIDAS PELAS LEIS E REGULAMENTOS FEDERAIS DOS ESTADOS UNIDOS DA AMÉRICA.

### Limites nas Interações

O pessoal governamental envolvido na competição MCTF não deverá adotar condutas que favoreçam um competidor em detrimento de outro, garantindo que todos os competidores tenham igualdade de oportunidade e acesso aos recursos.
