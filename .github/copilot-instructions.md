# DRIMA: A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting

## Project Overview

### Description

- DRIMA (**D**ynamics of **R**andomly **I**nteracting **M**oving **A**gents) is a minimal multi-agent system designed to study how reactive agents influence and adapt to one another’s behaviors through interactions;
- Implemented in Mathematica, it models agents moving on a 2D grid with periodic boundaries;
- Each agent occupies one cell and follows either deterministic (D-agents) or random (R-agents) movement patterns;
- When agents come within an interaction range, their movement probabilities are modified — mimicking attraction or influence between individuals;
- Over time, these interactions can lead to emergent patterns such as convergence of random agents toward deterministic ones;
DRIMA serves as a metaphorical model for social, behavioral, or conceptual adaptation in groups, emphasizing simplicity and emergent complexity.

### Technologies

- Python 3.12.2.

### Coding Standards

- Follow PEP 8 guidelines for Python code style;
- Use meaningful variable and function names;
- Write docstrings for all public modules, functions, and classes;
- Include at least one example usage in comments. Provide an overview of the project you're working on, including its purpose, goals, and any relevant background information.

## Coding and Explanation Instructions

### Explaining Technical Concepts

- Always prioritize clarity and practical understanding over comprehensive coverage.

#### Start Simple, Build Up

- Introduce technical terms gradually after concepts are clear;
- Build each new idea on what was already explained;
- Use concrete examples before abstract theory.

#### Make It Practical

- Include working code examples that demonstrate the concept;
- Connect theory to problems developers actually face;
- Provide step-by-step implementation when relevant.

#### Address Common Confusion

- Explain what NOT to do and why;
- Address edge cases that often cause problems;
- Show debugging approaches when things go wrong.

### Reviewing Code

#### Performance Red Flags

- Spot inefficient loops and algorithmic issues;
- Check for memory leaks and resource cleanup;
- Review caching opportunities for expensive operations.

#### Code Quality Essentials

- Functions should be focused and appropriately sized;
- Use clear, descriptive naming conventions;
- Ensure proper error handling throughout.

#### Review Style

- Be specific and actionable in feedback;
- Explain the "why" behind recommendations;
- Acknowledge good patterns when you see them;
- Ask clarifying questions when code intent is unclear.

## Project Business Rules

### Concepts

#### Grid

- Definido por uma grade discreta bidimensional onde há agentes que se movem nela;
- Cada agente ocupa uma única célula da grade do tipo autômato celular, que possui condições de contorno periódicas;
    - Ou seja, cada aresta da grade é envolvida pela sua aresta oposta.
- Mundo é um instantâneo do estado de todos os agentes do mundo, incluindo a posição atual na grade e padrão de movimento, bem como os valores de quaisquer outras propriedades que possam ter sido definidas para eles;
- História é a sequência de mundos gerados;
- Para representar o mundo, usamos a lista `{list_of_agents, grid_size, types_of_interaction, world_mobility}`;
- Os parametros sao auto-explicativos, mas para ir mais fundo:
    - O `world_mobility` eh um parametro definido pelo usuario que pode assumir os valores F ou P, representando, respectivamente, mobilidade total ou parcial;
        - Quando o mundo for F (Full Mobility) os agentes devem sempre se mover a cada iteração;
            - A probabilidade de “não se mover” (direção X) é fixada em 0.
        - E quando for P (Partial mobility) os agentes podem permanecer na mesma célula, ou seja, a probabilidade de “não mover” (X) é maior que 0 e variável.

#### Agents

- Os dois tipos de agentes são denominados D e R;
    - Os agentes D se movem deterministicamente;
    - Os agentes R se movem aleatoriamente.

##### Representation

- Todo agente eh representado pela lista `{identity, position}`, onde:
    - `identity = {name, movement_pattern, list_of_properties}`, eh um conjunto de caracteristicas que define o nome e o estado do agente;
        - O atributo `name` eh a string formada pela concatenacao do tipo do agente (D ou R) e o numero de identificacao unico (id);
        - O atributo `movement_pattern = {Pe, Pne, Pn, Pnw, Pw, Psw, Ps, Pse, X}`, onde Pd eh a probabilidade de o agente se mover na d-direcao (todas as posicoes de uma rosa dos ventos);
        - O atributo `list_of_properties` nao sera implementado.
    - E `position = {x, y}` da ao agente suas coordenadas atuais.

### Rules

#### Movement

- As possibilidades de movimentacao sao:
    - East (E), Northeast (NE), North (N), Northwest (NW), West (W), Southwest (SW), South (S), Southeast (SE) e X para "no-movement";
    - A forma como um agente se move eh denominada padrao de movimento, onde:
        - Agentes do tipo D possuem probabilidade 1 para alguma direcao e 0 para todas as outras;
        - Agentes do tipo R possuem probabilidades diferentes que definem a tendencia do seu movimento em qualquer direcao, incluindo a possibilidade de ficar parado;
        - A soma deve dar 1.

#### Interactions

- Os agentes podem interagir apenas de forma binária (dois por vez), e o resultado da interação entre dois agentes é uma mudança em seus padrões de movimento individuais;
- Se dois agentes escolhidos estiverem em uma distancia maior ou igual do **alcance de interacao** (que pode ser definida pelo usuario, mas por padrao eh igual a 1), podera haver interacao entre eles; caso for igual a 0, significa que as interações podem ocorrer quando ambos os agentes compartilham a mesma célula da grade.

##### Elements

- **Alcance de Ação / Interacao**: Usa a vizinhança de Moore com raio R (geralmente 1). A influência dos agentes é simétrica e sem direção preferencial. Se R = 0, a interação ocorre apenas entre agentes na mesma célula;
- **Escopo**: Pode ser binário ou múltiplo, mas atualmente apenas interações binárias (entre dois agentes) são permitidas;
- **Tipo**: Pode envolver agentes do mesmo tipo ou de tipos diferentes. O usuário define quais tipos são afetados ou permanecem inalterados;
- **Modo**: Pode ser síncrono (todos os agentes interagem em sequência, com ordem aleatória a cada iteração) ou assíncrono (um agente e um vizinho aleatórios interagem por vez);
    - No modo **SINCRONO**, todos os agentes são processados em sequência dentro da mesma iteração;
        - Cada agente interage uma vez por rodada, com um vizinho escolhido aleatoriamente;
        - A ordem é embaralhada a cada iteração para evitar viés.
    - No modo **ASSINCRONO**, a cada passo é escolhido aleatoriamente um agente e um vizinho aleatório dentro do alcance; eles interagem, e o estado do mundo é atualizado imediatamente;
        - O processo se repete várias vezes.
- **Direção**: Unidirecional (apenas um agente é afetado) ou bidirecional (ambos são afetados). Atualmente, apenas o modo unidirecional é implementado;
- **Efeito**: Direto (o movimento do agente muda imediatamente após a interação) ou indireto (efeito mediado por outra variável);
- **Intensidade**: Define o quanto algo muda na interação — pode ser fixa ou variável, escolhida entre valores discretos predefinidos;
- **Determinismo**: As interações são determinísticas, embora os movimentos dos agentes sejam não determinísticos;
- **Conteúdo**: Reativo, ou seja, o resultado depende apenas do estado atual dos agentes; não há memória nem representação interna do mundo.

##### Types

- O tipo de interacao deve estar contido na lista `{dd, DD, rr, RR, dR, Rd}`;
- Se o tipo de agente for especificado em letras maiúsculas, isso significa que ele sofrerá uma interação efetiva; ou seja, seu padrão de movimento poderá ser alterado pela interação;
- Por outro lado, se o tipo de agente for especificado em letras minúsculas, a interação se torna ineficaz para o agente, pois ele sempre sairá inalterado;
- Por exemplo: enquanto DD implica que ambos os agentes podem ser modificados pela interação, dd implica que eles nunca serão;
- E embora as interações Dr ou rD nao serao implementadas;
- As interacoes padrao sao `{dd, RR, dR, Rd}`.

##### The Direct Interactions

###### General Aspects

- Cada agente possui um padrão de movimento: um conjunto de probabilidades associadas às 8 direções possíveis (N, NE, E, SE, S, SW, W, NW) e à possibilidade de não se mover (X);
- Quando dois agentes interagem, ocorre uma alteração local nessas probabilidades, em cada direção, com base nos valores atuais de ambos;
- A interação é direta, isto é, **modifica imediatamente o padrão de movimento**;
- Apenas o menor valor de probabilidade do par (entre os dois agentes) é aumentado;
- Após a atualização, o vetor de probabilidades é **normalizado** para que a soma total continue igual a 1.

###### Fixed Mode

- A variação da probabilidade é constante, independente dos valores atuais;
- Fórmula: `\Delta p = Fp/9`, onde:
    - `Fp`: fator de incremento de probabilidade;
    - `9`: número de direções possíveis (8 + 1 sem movimento);
    - Quanto maior o Fp, mais rápidas e instáveis se tornam as mudanças;
        - Exemplo: Fp = 1 → convergência estável; Fp = 5 → instabilidade e oscilações.

###### Variable Mode

- A variação depende da diferença entre as probabilidades atuais dos dois agentes em uma mesma direção;
- Fórmula: `\Delta p = (P_d^i - P_d^j) \times \frac{Fp}{9}`, onde:
    - `P_d^i` e `P_d^j`: probabilidades dos agentes i e j naquela direção;
- Assim, quanto maior a diferença entre eles, maior o impacto da interação;
- Mantém o mesmo Fp como fator de escala, mas o valor final de Δp muda dinamicamente.


###### Other Rules About Fixed / Variable Modes

- O aumento nunca pode tornar a menor probabilidade maior que a maior probabilidade pré-interação;
- Após cada atualização, é feita normalização:
    - Se a soma > 1: subtrai-se igualmente o excesso de todas as direções com probabilidade > 0;
    - Se a soma < 1: adiciona-se igualmente o déficit.
- Probabilidades negativas são truncadas a 0, e o processo pode ser iterado até estabilizar.
