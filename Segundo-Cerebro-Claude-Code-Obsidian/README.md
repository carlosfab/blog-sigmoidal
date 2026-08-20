# Segundo Cérebro com Claude Code e Obsidian

Este material ajuda você a criar uma base de conhecimento pessoal mantida por um agente de IA. O Obsidian funciona como interface de leitura e navegação; o Claude Code organiza, relaciona e atualiza o conhecimento armazenado em arquivos Markdown.

O ponto de partida é um prompt de configuração. Antes de gerar qualquer arquivo, o agente entrevista você para entender o domínio da base, os tipos de fonte, a estrutura desejada e o nível de automação. Em seguida, ele propõe a arquitetura e entrega as instruções necessárias para operar a wiki.

## Material

- [Abrir o prompt para construir seu segundo cérebro](PROMPT.md)
- [Referência conceitual: LLM Wiki, por Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

## Como usar

1. Crie ou escolha uma pasta vazia para a sua base de conhecimento.
2. Abra o Claude Code nessa pasta.
3. Copie todo o conteúdo de [`PROMPT.md`](PROMPT.md) e envie ao agente.
4. Responda às perguntas de configuração.
5. Revise a arquitetura proposta antes de autorizar a criação dos arquivos.
6. Abra a pasta como um *vault* no Obsidian depois que a estrutura estiver pronta.

O prompt também pode ser adaptado para outros agentes de programação que operem diretamente sobre arquivos locais.

## Estrutura conceitual

```text
fontes brutas e imutáveis
          ↓
      ingestão
          ↓
wiki em Markdown interligada
          ↓
consultas, sínteses e auditorias
          ↓
conhecimento novo incorporado à wiki
```

O princípio central é simples: **as fontes continuam preservadas, enquanto a wiki se torna um artefato cumulativo, versionável e progressivamente mais útil**.

