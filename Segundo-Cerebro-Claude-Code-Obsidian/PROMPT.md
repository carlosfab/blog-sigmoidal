# Prompt: construa meu segundo cérebro com Claude Code e Obsidian

Copie o conteúdo abaixo e envie ao Claude Code aberto na pasta que será usada como raiz da sua base de conhecimento.

---

Quero construir uma base de conhecimento pessoal em Markdown, mantida por você e navegada por mim no Obsidian. A base deve funcionar como uma wiki cumulativa: novas fontes não serão apenas armazenadas ou recuperadas sob demanda; elas deverão ser lidas, sintetizadas e integradas ao conhecimento existente.

Você atuará como arquiteto e mantenedor dessa base. Sua primeira tarefa não é criar arquivos. Antes de qualquer implementação, conduza uma entrevista curta para calibrar o sistema ao meu contexto.

## Fase 1 — Entrevista obrigatória

Faça apenas as perguntas que realmente alterem a arquitetura ou o fluxo de trabalho. Agrupe-as em uma única rodada sempre que possível. Cubra, no mínimo:

1. Qual é o objetivo principal da base de conhecimento?
2. Quais domínios ou assuntos ela deverá cobrir?
3. Quais tipos de fonte serão adicionados: artigos, PDFs, papers, livros, notas, transcrições, repositórios, datasets, imagens ou outros?
4. Já existe conteúdo na pasta? Em caso afirmativo, o que pode e o que não pode ser reorganizado?
5. O Obsidian será apenas uma interface de leitura ou também será usado para edição manual?
6. Preciso de propriedades YAML, Dataview, tags, aliases ou outro recurso específico do Obsidian?
7. Qual deve ser o idioma das páginas, dos nomes de arquivo e das instruções operacionais?
8. Como devo tratar informações sensíveis, privadas ou que não podem ser enviadas a serviços externos?
9. Quero processar as fontes uma a uma, em lotes ou das duas formas?
10. Quais saídas pretendo gerar a partir da wiki: respostas, relatórios, artigos, apresentações, mapas conceituais, gráficos, código ou outras?
11. Desejo versionamento com Git? Quais arquivos devem ser ignorados?
12. Há ferramentas locais que você pode utilizar ou devo começar somente com arquivos Markdown e comandos básicos?

Se uma resposta revelar outra decisão estrutural indispensável, faça uma segunda rodada curta. Não prolongue a entrevista com preferências cosméticas que possam ser definidas depois.

## Fase 2 — Proposta antes da implementação

Depois das minhas respostas, apresente uma proposta concisa contendo:

- objetivos e limites do sistema;
- árvore de diretórios;
- tipos de página e responsabilidades de cada um;
- esquema de metadados;
- convenções de nomes, links e citações;
- fluxo de ingestão;
- fluxo de consulta e síntese;
- fluxo de auditoria e manutenção;
- decisões que você poderá tomar autonomamente;
- decisões que exigirão minha aprovação;
- ferramentas opcionais e o momento em que passariam a ser justificadas.

Espere minha aprovação antes de criar, mover, renomear ou excluir arquivos.

## Princípios arquiteturais obrigatórios

A proposta deve respeitar os princípios abaixo, adaptando os detalhes às minhas respostas.

### 1. Fontes brutas são imutáveis

Mantenha as fontes originais em `raw/` ou em outro diretório claramente identificado. Você pode ler, catalogar e referenciar esses arquivos, mas nunca deve editar seu conteúdo original. Conversões necessárias devem gerar novos arquivos derivados sem substituir a fonte.

### 2. A wiki é um artefato compilado e cumulativo

Mantenha as páginas produzidas pelo agente em `wiki/`. Ao incorporar uma nova fonte, atualize as páginas conceituais, entidades, comparações e sínteses afetadas. Evite criar uma coleção de resumos isolados que não se relacionam entre si.

### 3. Toda afirmação verificável deve preservar sua origem

Registre referências de forma rastreável até o arquivo em `raw/` e, quando disponível, até a página, seção, timestamp, URL ou identificador específico. Diferencie claramente:

- fatos sustentados pelas fontes;
- declarações ou opiniões atribuídas a terceiros;
- interpretações e inferências produzidas pelo agente;
- hipóteses que ainda precisam de verificação.

Nunca preencha lacunas factuais com conteúdo inventado.

### 4. Contradições não devem ser ocultadas

Quando duas fontes discordarem, registre a divergência, a data e o contexto de cada afirmação. Não escolha silenciosamente uma versão. Proponha um critério de resolução e peça orientação quando a decisão depender de julgamento humano.

### 5. A navegação deve permanecer legível

Use links internos, backlinks ou seções de páginas relacionadas de maneira consistente. Evite páginas órfãs. Crie uma nova página quando o conceito tiver identidade, evidência ou relações suficientes para justificar manutenção própria; caso contrário, atualize uma página existente.

### 6. Comece simples

Não adote embeddings, banco vetorial, RAG, servidor, interface própria ou automações complexas antes que o volume e os padrões de uso demonstrem necessidade. Inicialmente, priorize arquivos Markdown, índices claros e busca local. Qualquer ferramenta adicional deve resolver um problema observado e mensurável.

### 7. Mudanças devem ser auditáveis e recuperáveis

Registre as operações relevantes em um log cronológico. Quando Git estiver habilitado, mantenha alterações pequenas e coerentes. Nunca execute exclusões, reestruturações amplas ou sobrescritas difíceis de recuperar sem minha aprovação explícita.

## Artefatos mínimos

Depois que eu aprovar a proposta, crie os artefatos adequados ao agente em uso e ao meu contexto. A implementação deve incluir, no mínimo:

### Arquivo de instruções do agente

Crie `CLAUDE.md` para Claude Code. Se eu estiver usando outro agente, proponha o arquivo equivalente, como `AGENTS.md`, sem manter dois documentos concorrentes com regras divergentes.

Esse arquivo deve explicar:

- finalidade e escopo da base;
- arquitetura de diretórios;
- fonte de verdade de cada camada;
- esquema e convenções das páginas;
- regras de autonomia e aprovação;
- tratamento de fontes, citações e contradições;
- procedimentos de ingestão, consulta e auditoria;
- verificações obrigatórias antes de encerrar cada operação.

### Índice

Crie um `index.md` orientado ao conteúdo. Ele deve permitir que uma pessoa ou agente localize rapidamente as páginas relevantes sem ler toda a wiki. Para cada página, inclua link e descrição breve. Organize o índice por categorias coerentes com o domínio.

### Log

Crie um `log.md` cronológico e somente aditivo. Use cabeçalhos padronizados e fáceis de pesquisar, por exemplo:

```markdown
## [AAAA-MM-DD] ingest | Título da fonte
## [AAAA-MM-DD] query | Pergunta ou análise
## [AAAA-MM-DD] lint | Escopo da auditoria
```

Cada entrada deve registrar o que mudou, quais arquivos foram afetados, quais decisões foram tomadas e quais pendências permanecem abertas.

### Templates

Crie apenas os templates necessários ao domínio. Um template de página deve, quando pertinente, prever:

- título e aliases;
- tipo da página;
- status de revisão;
- datas de criação e atualização;
- fontes relacionadas;
- resumo;
- afirmações principais com evidências;
- divergências ou incertezas;
- relações com outras páginas;
- perguntas em aberto.

## Operações que a wiki deve suportar

### Ingestão

Ao receber uma nova fonte:

1. verifique se ela já foi processada;
2. preserve o original em `raw/`;
3. identifique metadados, escopo, confiabilidade e limitações;
4. produza ou atualize o registro da fonte;
5. integre o conhecimento às páginas relevantes;
6. crie novas páginas somente quando justificadas;
7. atualize links, referências e índice;
8. registre a operação no log;
9. apresente um resumo das mudanças, divergências e perguntas abertas.

### Consulta

Ao responder a uma pergunta:

1. consulte primeiro o índice;
2. localize e leia as páginas e fontes relevantes;
3. diferencie evidência, interpretação e incerteza;
4. cite as fontes utilizadas;
5. informe quando a wiki não contém evidência suficiente;
6. proponha pesquisa adicional somente quando necessária;
7. pergunte se uma síntese nova e reutilizável deve ser incorporada à wiki.

Não transforme automaticamente toda resposta em uma página permanente.

### Auditoria

Ao executar uma auditoria da wiki, verifique:

- páginas órfãs;
- links quebrados;
- referências ausentes;
- afirmações sem evidência;
- duplicações e sobreposição de escopo;
- contradições não registradas;
- conteúdo possivelmente desatualizado;
- conceitos importantes sem página própria;
- metadados incompletos;
- arquivos brutos ainda não processados;
- páginas que deveriam ser consolidadas ou divididas.

Apresente o diagnóstico antes de realizar mudanças estruturais amplas.

## Critérios de qualidade

A implementação final deve ser:

- compreensível sem depender desta conversa;
- simples o suficiente para começar a usar imediatamente;
- específica ao domínio e às respostas da entrevista;
- compatível com Markdown e Obsidian;
- rastreável até as fontes;
- resistente a duplicações e contradições silenciosas;
- segura para evolução incremental;
- documentada com exemplos curtos dos principais fluxos.

Comece agora pela entrevista da Fase 1. Não crie arquivos ainda.

