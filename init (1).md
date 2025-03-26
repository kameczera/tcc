#### Disclaimer:

Qualquer projeto que envolva o pytorch vai requerer um conhecimento profundo do pytorch, e de análises de fluxo.
Um projeto de mera "comparação" não é o suficiente. Queremos algo novo, uma contribuição relevante.



### Ideia de Projeto 

#### 4. pytorch + numba: Improved pytorch.compile() through number 
Nível de dificuldade: alto
Recompensa: baixa + aprendizado alto

Inserir o numba no pytorch. Isso já foi feito? Em qual fase do pipeline ele entraria?

Passos:
-> Estudar IR's e análises de fluxo do pytorch.
-> Estudar a IR do torchinductor (torch.compile()).
-> Estudar o numba.
-> Forkar o pytorch.
-> Estudar como o numba pode ser inserido no pytorch.

Contribuição: 
Ferramenta que decide quando usar numba no torch.compile() e quando não usar. Seria uma feature do pytorch, que pode entrar via Pull Request se estiver madura e coberta por testes.    

_Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code._
https://numba.pydata.org/

#### 3. Paralelismo automático no grafo de dependências de instrução (CPU)
Nível de dificuldade: médio
Recompensa: média

É possível encontrarmos oportunidades de paralelismo no grafo de dependências a nível de instrução?

-> Paralelismo em CPU, mas partes de cópia/transferência de dados sequenciais para a GPU, por exemplo, nesses modelos.

Contribuição: Ferramenta que encontra parelelismo automático em python puro executando em CPU. Não é necessariamente relacionado a machine learning.

#### 2. Por que algumas fusões não são capturadas pelo torch.compile()?
Nível de dificuldade: médio
Recompensa: média

```python
x = torch.randn(10, 10)
y = torch.sin(x) + torch.cos(x)
```

O torchinductor vai mesclar as operações de sin e cos acima? 

Contribuição:
Ferramenta que aponta quantas/quais operações não foram mescladas após o operator fusion isolado do pytorch. 

Roadmap:
Introdução: Operator Fusion, pytorch2, torchinductor vs torchscript. [refs ok]
Justificativa: Nem todas as operações são fundidas pelo torchinductor. [refs !!!]
Proposta (TCC): Identificar quais operações não são fundidas os benchmarks do pytorch e o motivo delas.
Identificar quais, e quantas e se poderiam ou não serem fundidas.

==========================================

Trabalhos futuros: 
Escrever um novo passe para o torchinductor que realiza fusões que antes ele não realizava. 


#### 1. Operator Fusion com FX vs torch.compile()
Nível de dificuldade: alto
Recompensa: baixa + aprendizado alto

Escrever uma transformação de fusão de operações com tensores e comparar com a performance do torch.compile() (que utiliza o torchinductor). 

Será desafiador pois não será trivial isolar o operator fusion do torchinductor. 

`torch.compile()` aceita opções como "reduce-overhead" mas ainda assim ele executa algumas otimizações além de fusão de operadores. Podemos inicialmente testar com essa flag.

Porém, é esperado que o torchinductor apresente performance superior. 

Caso seja esse o cenário, podemos "tweak" o torchinductor para encontrar oportunidades de melhoria.

Podemos abrir um fork() do pytorch e fazer as alterações necessárias para que ele execute somente o operator fusion.

Obs: pelo menos 1 otimização do torch.compile() precisamos eliminar definitivamente: paralelização automática, pois não faremos isso no nosso passe customizado.

Mediremos com `torch.profiler` o tempo de execução do modelo em CUDA.
https://pytorch.org/docs/stable/profiler.html

Requisito: 
* CUDA instalado + placa de vídeo NVIDIA.

##### Caso base: 

Modelo executando sem nenhuma otimização.

##### Passe customizado:

Modelo executando somente com o passe de fusão de operadores escrito com o FX usando symbolic tracing.

##### Passe tradicional:

Modelo executando com o passe de fusão de operadores do torchinductor. Será desafiador isolar o operator fusion na execução do torch.compile().

### Tutorial de como escrever uma transformação com FX:
https://pytorch.org/docs/main/torch.compiler_transformations.html

### Código fonte da pasta que teríamos que alterar para isolar o passe de fusão de operadores:
https://github.com/pytorch/pytorch/tree/main/torch/_inductor/fx_passes

### Code-generator do inductor
https://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen

### Interface do torch.compile()
https://pytorch.org/docs/stable/generated/torch.compile.html

### FX Documentation
https://pytorch.org/docs/main/fx.html#torch-fx

### Escrevendo transformações usando FX
https://pytorch.org/docs/main/fx.html#writing-transformations