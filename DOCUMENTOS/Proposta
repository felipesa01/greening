# USO DE IMAGENS DE VANT PARA IDENTIFICAÇÃO E

# MONITORAMENTO DA DOENÇA DE CITROS

# HUANGLONGBING (GREENING) EM LAVOURAS DE LARANJA

# DO CINTURÃO CITRÍCOLA

Felipe Rafael de Sá Menezes Lucena

```
Proposta de Dissertação de Mestrado
do Curso de Pós-Graduação em
Sensoriamento Remoto, orientada pelo
Dr. Hermann Johann Heinrich Kux.
```
#### INPE

```
São José dos Campos
```

```
ii
```
#### RESUMO

O cultivo de laranja é uma das mais importantes atividades agrícolas do Brasil
e, além de ser responsável pelo abastecimento nacional, contribui fortemente
com a economia do país através da exportação da produção. Contudo, apesar
de ser o maior produtor de laranja do mundo, o Brasil vem diminuindo a sua
capacidade produtiva ao longo dos anos devido aos problemas que impactam a
sustentabilidade econômica do cultivo. Um dos principais problemas
enfrentados pelos produtores é a necessidade de controle e manejo de
problemas fitossanitários, como as doenças que assolam a lavoura. O
_Huanglongbing_ é uma doença dos citros que impacta diretamente na
produtividade das árvores infectadas e sua identificação precoce é
indispensável para conter a disseminação dos danos. O uso de sensoriamento
remoto para identificação de plantas doentes em lavouras de laranja é um
importante aliado uma vez que as outras formas de identificação demandam
grandes custos operacionais ou são inviáveis para monitoramento contínuo da
doença. Nessa pesquisa será avaliada a capacidade do uso de imagens de
sensoriamento remoto de altíssima resolução espacial e algoritmos de
aprendizagem de máquina para identificação e monitoramento da doença em
talhões comerciais de laranjeiras. Com isso, espera-se contribuir com o manejo
da fitopatologia no país frente ao problema de redução crônica da produção do
fruto.

Palavras-chave: Sensoriamento remoto. VANT. Identificação de doenças.
_Machine learning. Deep Learning_


## iii

- 1. INTRODUÇÃO SUMÁRIO
   - 1.1. Objetivos
   - 1.2. Objetivos específicos
- 2. FUNDAMENTAÇÃO TEÓRICA
   - 2.1. Citricultura no Brasil
   - 2.2. Doença de Huanglongbing
   - 2.3. Uso de sensoriamento remoto no monitoramento agrícola
   - 2.3.1. Sensoriamento remoto orbital e aéreo
   - 2.3.2. Técnicas de processamento digital de imagens
- 3. MATERIAIS E MÉTODOS
   - 3.1. Área de estudo
   - 3.2. Dados
   - 3.2.1. Imagens VANT
   - 3.2.2. Inspeção visual
   - 3.3. Metodologia
   - 3.3.1. Pré-processamento
   - 3.3.2. Extração de atributos
   - 3.3.3. Segmentação
   - 3.3.4. Classificação
- 4. RESULTADOS ESPERADOS
- 5. CRONOGRAMA


#### 1. INTRODUÇÃO

O cultivo de laranjas é uma das principais atividades da citricultura mundial e,
nacionalmente, corresponde à maior produção de frutos do país. Além disso, o
Brasil é o maior produtor da fruta no cenário internacional e ocupa a segunda
posição – atrás da China – no cultivo de citros em geral (FAO, 2019). A
atividade impacta diretamente na dinâmica socioeconômica nacional, tendo
sido responsável, em 2009, pela criação de 230 mil novos empregos diretos e
indiretos (NEVES, 2010). Apenas no estado de São Paulo, foram geradas 48
mil vagas de trabalho em 2019, o que corresponde a 26% das vagas abertas
no estado (SÃO PAULO, 2020).

O cinturão citrícola, localizado no sudoeste brasileiro, é a região do país com a
maior produção de citros, sobretudo da laranja, e é responsável por quase 80%
da produção da fruta (FUNDECITRUS, 2020a). Entretanto, a região vem sendo
prejudicada por fatores que afetam a sustentabilidade econômica da produção,
como o aparecimento e a disseminação de problemas fitossanitários, sendo o
principal deles o Huanglongbing.

O Huanglongbing (HLB, do chinês “doença do ramo amarelo”), também
conhecido como _greening,_ é considerado a doença mais importante para a
citricultura mundial dado o desafio para o manejo e atenuação dos seus
impactos (BOVÉ, 2006). De acordo com Rodrigues et al. (2020), a identificação
prévia da doença nas lavouras de laranjas é de extrema importância, uma vez
que favorece a redução do potencial de disseminação e diminui os prejuízos
financeiros da produção.

As formas de detecção da doença mais utilizadas atualmente apresentam uma
série de desvantagens relacionadas ao tempo de resposta, custos elevados,
necessidade de mão de obra especializada e/ou precisão insuficiente do
diagnóstico (FUTCH; WEINGARTEN; IREY, 2009; JUNIOR et al., 2010; DENG
et al., 2020). Diante deste cenário, o sensoriamento remoto surge como uma
alternativa viável de detecção doenças em lavouras agrícolas (JUNG et al.,
2021).


No contexto do sensoriamento remoto, diversas plataformas são utilizadas para
imageamento da superfície terrestre e suas aplicações dependem do objeto de
estudo analisado. Os Veículos Aéreos não Tripulados (VANTs), são um tipo de
plataforma de nível aéreo capaz de, com uso de sensores espectrais, realizar o
imageamento de talhões agrícolas com características adequadas –
mencionadas na seção 2.3 – para a detecção de doenças agrícolas.

Uma variedade de abordagens envolvendo a detecção de doenças com uso de
VANTs vêm sendo estudadas ao longo dos últimos anos (JR; DAUGHTRY,
2018) e os resultados indicam o grande potencial de detecção aliado à
viabilidade de implementação de sistemas de monitoramento a partir do
sensoriamento remoto. Entretanto, o monitoramento do _greening_ nos estados
de São Paulo e Minas Gerais ainda é realizado por meio da inspeção visual e
individualizada das árvores (FUNDECITRUS, 2020c), o que demanda grandes
recursos para execução da atividade.

Com o uso de imagens de sensoriamento remoto, diversas técnicas de
processamento digital podem ser aplicadas. Dentre elas, as técnicas de
_machine learning_ e _deep learning_ ganharam muita importância com o
desenvolvimento da computação e do aprendizado de máquina. Estudos
indicam que a detecção de doenças em plantas com uso de imagens de
sensoriamento remoto, aéreo ou terrestre, e de abordagens de aprendizado de
máquina atinge os melhores resultados e são extremamente viáveis para a
atividade (LAN et al., 2020; OSCO et al., 2021).

Diante disso, é necessária a investigação da capacidade de detecção e
monitoramento do HLB em plantações de laranja do cinturão citrícola, de modo
que sejam explorados procedimentos diagnósticos mais eficientes do que os
utilizados atualmente e sejam reduzidos os custos da produção, aumentando a
sustentabilidade econômica da atividade.

### 1.1. Objetivos

O estudo pretendido por essa proposta de dissertação objetiva avaliar a
capacidade de identificação e monitoramento de laranjeiras de talhões
comerciais contaminadas pelo HLB com o uso de imagens captadas por VANT.
A pesquisa irá contemplar a análise do potencial de identificação e


discriminação das árvores sintomáticas por meio de técnicas de _deep learning_ ,
bem como da viabilidade de implementação de um sistema contínuo de
monitoramento da doença.

### 1.2. Objetivos específicos

Realizar a identificação e delimitação de modo automatizado das copas das
árvores na área de estudo.

Identificar o conjunto de atributos espaciais que melhor discriminam o evento
de contaminação por HLB nas árvores identificadas e delimitadas.

Avaliar a capacidade de detecção das plantas infectadas de acordo com os
diferentes níveis de severidade da doença.

Avaliar a possibilidade da implantação de um sistema de monitoramento da
doença a partir de um _framework_ que abrange todos os processamentos
realizados.


## 2. FUNDAMENTAÇÃO TEÓRICA

### 2.1. Citricultura no Brasil

A citricultura compreende o cultivo de laranja, lima, limão, tangerina e _grapefruit_
e é uma das principais atividades agrícolas do Brasil. Além da produção para
abastecimento do mercado interno, o setor apresenta significativa importância
para a economia nacional e foi responsável, em 2010, pela exportação de 26%
da produção mundial de laranja e por 56% de todo suco de laranja consumido
mundialmente (NEVES et al., 2010). Atualmente, apesar da variação na
produção, o país é um dos maiores produtores de citros e mantém sua posição
de destaque no cenário internacional (FAO, 2019).

O cultivo dos citros foi iniciado ainda no período da colonização devido à
disponibilidade de condições favoráveis para o plantio e se expandiu por todo
território nacional. Desde a década de 1960, com a exportação do suco de
laranja, a produção comercial de citros se intensificou no país e, ao longo dos
anos, o Brasil tornou-se o principal produtor de laranjas do mundo (BOTEON,
M.; NEVES, 2005; FAO, 2019).

Desde então, a laranja é a fruta mais cultivada dentre os citros, representando
87,3% da produção em 2019 , e é ainda a fruta mais cultivada no país, com
mais de 17 milhões de toneladas produzidas. Considerando todos os produtos
agrícolas, seja de lavouras temporárias ou permanentes, a laranja ocupa a
quinta posição na produção nacional, precedida pela Cana-de-açúcar, Soja,
pelo Milho e pela Mandioca. (IBGE, 2019 ).

Apesar de estar presente em todos os estados brasileiros, a maior parte do
cultivo de citros, sobretudo da laranja, está concentrada espacial e
potencialmente no estado de São Paulo, que é responsável por 63,97% da
área colhida e por 77,64% da produção total da fruta no país (IBGE, 2019).

Além de São Paulo, o triângulo/sudoeste mineiro também é um potencial
produtor citrícola e a região formada pelos 374 principais municípios dos dois
Estados corresponde ao chamado cinturão citrícola (Figura 1 ), região de maior
produção do setor no País. (FUNDECITRUS, 2020).


```
Figura 1 – Localização do cinturão citrícola
```
Fonte: Adaptado de Fundecitrus (2020b).

O cinturão é dividido em cinco setores denominados norte, sul, centro,
sudoeste e noroeste, que agrupam as chamadas regiões. As regiões
correspondem ao agrupamento dos municípios que as compõem e recebem,
com exceção da região Triângulo Mineiro, o nome de um deles.

Segundo Franco (2016), o cinturão citrícola é contemplado com uma série de
fatores ambientais e socioeconômicos que favorecem o seu destaque na
produção de laranjas. Quanto aos aspectos ambientais, a autora pontua o
padrão topográfico, a presença de solos férteis e o clima favorável como as
principais vantagens para a produção na região.

Em relação aos fatores socioeconômicos, os principais benefícios encontrados
são a mão de obra qualificada, a disponibilidade de insumos e a presença e
atuação de diversos institutos de pesquisa no monitoramento das lavouras e
acompanhamento aos produtores. Esses institutos visam encontrar soluções
voltadas ao combate de perdas na produção e à melhoria da qualidade das
frutas (FRANCO, 2016).


Os esforços dos agentes envolvidos no cultivo de citros e as condições
ambientais e socioeconômicas favoráveis subsidiam, portanto, o destaque do
cinturão e do Brasil no plantio desse setor. Entretanto, apesar da grande
produção, a região vem sofrendo redução da sua capacidade produtiva nos
últimos anos (FUNDECITRUS, 2020b), o que demanda atenção na gestão
agrícola de modo a garantir a sustentabilidade da citricultura.

No levantamento realizado por Erpen et al. (2018), a análise do cultivo de
laranja entre os anos de 2001 e 2015 detectou redução de 9,2% da produção
no cinturão citrícola. Considerando as estimativas para a safra de 2020- 2021 , a
redução projetada é de 12,5% em relação à média dos últimos dez anos e de
25,6% à safra anterior (FUNDECITRUS, 2020a), indicando a constância na
queda da produção durante os últimos 20 anos.

Ainda segundo Erpen et al. (2018), a área de cultivo da laranja foi a variável
mais impactada na região do cinturão entre 2001 e 2015, com redução de 29%,
sendo a maior parte dessas terras transformadas em cultivo de cana-de-
açúcar. Nos últimos anos, a área destinada ao plantio de laranja segue em
declínio e teve redução de 1,44% na estimativa de 2021 em relação a 2019 –
de 401,47 ha para 395,67 ha (FUNDECITRUS, 2020b).

Uma das maiores razões para a redução observada é o aumento nos custos de
produção da cultura, que tem como principais componentes os gastos com
mão de obra e com o controle fitossanitário (PAGLIUCA, 2012; ERPEN et al.,
2018). Um estudo amostral envolvendo 272 propriedades produtoras de laranja
do cinturão citrícola demonstrou (Figura 2 ) que, segundo os agricultores, a
principal dificuldade para produzir laranjas nas safras de 2017 a 2019 consiste
nos problemas fitossanitários que assolam a lavoura (NETO, 2019).


```
Figura 2 – Principais dificuldades encontradas pelos produtores no cultivo de laranja entre as
safras de 2017 e 2019.
```
Fonte: Neto (2019).

De acordo com Rossi (2017), a incidência de pragas e doenças no plantio de
laranjas é um problema substancialmente sério enfrentado pelos produtores e
tem como principais impactos a necessidade de contratação de mão de obra
especializada, o uso de produtos químicos para controle das doenças e a
execução de erradicações obrigatórias^1 , o que justifica o aumento nos custos
da produção.

Diante do exposto, é notório o destaque do cinturão citrícola e do país na
produção de citros, sobretudo da laranja. Por outro lado, este setor agrícola
enfrenta uma redução crônica na sua capacidade produtiva, principalmente
relacionada à perda de área cultivada gerada por fatores que afetam a
sustentabilidade econômica da produção, como o controle fitossanitário.

Dentre as principais doenças relacionadas à perda de produção de laranjas, é
possível destacar, por critérios de nocividade e porcentagem de incidência, o
Huanglongbing (HLB) ou _greening_ , o CVC e o cancro cítrico (FUNDECITRUS,
2019). O HLB é considerado pela literatura e pelos produtores como a principal

#### 1

As erradicações obrigatórias são aplicadas através das diretrizes da Instrução Normativa nº
53 de 16/10/2008 / MAPA - Ministério da Agricultura, Pecuária e Abastecimento
(D.O.U. 17/10/2008)


doença de citros da atualidade e tem demandado muito recurso financeiro e
humano no combate aos impactos causados (JUNIOR et al., 2009).

Sendo assim, na seção 2.2, será explorado um conjunto de informações a
respeito da doença e do controle necessário para amenizar as perdas geradas.

### 2.2. Doença de Huanglongbing

O HLB é uma doença causada, no Estado de São Paulo, pelas bactérias
_Candidatus_ Liberibacter asiaticus e _Ca._ L. americanus e a sua transmissão está
associada ao psilídeo dos citros _Diaphorina citri_ Kuwayama ( _D. citri_ ) (CAPOOR
_et al_ ., 1967; TEIXEIRA _et al_ ., 2005; JUNIOR _et al_ ., 2009).

O impacto causado nas plantas infectadas abrange tanto a estrutura das
folhas, ramos e frutos quanto a produtividade da laranjeira. É possível observar
a presença da doença a partir dos sintomas como amarelamento das folhas
dos ramos infectados, acúmulo de amido no tecido foliar, má formação e
permanência da coloração verde nos frutos maduros. Entretanto, Lin et al.
(2010) destacam que os sintomas observados podem ser, ainda, confundidos
com outros problemas bióticos e abióticos relacionados à saúde do vegetal.

Além dos sintomas visuais, o _greening_ pode afetar diretamente a produtividade
da planta. Segundo Fundecitrus, (2020c), à medida que a severidade dos
sintomas aumenta, a produção diminui, uma vez que reduz o crescimento e
causa a queda prematura dos frutos. O impacto na produção pode provocar
uma redução de até 100%, tornando o vegetal totalmente improdutivo.
(BASSANEZI, 2006).

O Fundo de Defesa da Citricultura (Fundecitrus), instituição que realiza o
monitoramento e controle do _greening_ junto aos produtores do parque citrícola,
classifica a severidade da doença em quatro níveis baseados na porcentagem
de cobertura dos sintomas na copa das árvores. O nível 1 representa as
árvores com até 25% da copa infectada sintomaticamente, no nível 2, de 26% a
50% da copa apresenta sintomas, nível 3, de 51% a 75% e o nível 4, de 76% a
100% (FUNDECITRUS, 2020c).


O grande desafio provocado pela doença consiste na condição de ainda não
haver uma abordagem curativa para as plantas infectadas. Desta forma, o
manejo da doença deve ser focado em procedimentos preventivos que visem
evitar novas contaminações, que são executados através da eliminação de
plantas sintomáticas e da aplicação de defensivos agrícolas para controle do
vetor (BOVÉ, 2006; BELASQUE ET AL., 2010).

O HLB foi descoberto na China no final do século XIX (DA GRAÇA, 1991) e,
apesar de ser uma doença centenária, foi relatada pela primeira vez no
continente americano apenas no século XXI, atingindo as principais regiões
produtoras de citros do mundo - os Estados de São Paulo, no Brasil, em 2004,
e da Flórida, nos EUA, em 2005. (COLETTA-FILHO _et al_ ., 2004; TEIXEIRA _et
al_ ., 2005).

Após sua identificação, o _greening_ rapidamente se disseminou por todas as
regiões citrícolas do estado e do Brasil, tendo sido constatada em 2005 em
Minas Gerais e em 2007 no Paraná (JUNIOR et al., 2009). A disseminação da
doença no cinturão citrícola ocorreu de forma acelerada, de modo que em 2004
apenas 3,4% dos talhões comerciais apresentavam alguma contaminação e
oito anos depois, em 201 2 , atingiu a maioria ( 64 %) das unidades produtoras
(ROSSI, 2017).

A partir de 2011, não foram encontrados dados a respeito da contaminação em
relação aos talhões, contudo, o monitoramento anual de doenças estima a
quantidade de plantas infectadas e na Figura 3 é possível observar os dados a
respeito da quantidade de árvores contaminadas no cinturão citrícola entre os
anos de 2008 e 2020.


```
Figura 3 – Percentual de árvores contaminadas pelo HLB no cinturão citrícola
```
Fonte: Fundecitrus (2020c)

Percebe-se que até 2009 o percentual de árvores contaminadas, apesar de
crescente, se manteve abaixo de 1% e teve seu crescimento acelerado a partir
de 2010, chegando ao índice atual de 20,87% das árvores com algum grau de
contaminação, que é definido pela severidade da doença na planta.

Segundo Bassanezi et al. (2020), o aumento significativo do número de árvores
contaminadas a partir de 2010 foi provocado principalmente pela descrença
dos produtores em relação à prática de erradicação das plantas sintomáticas
como uma forma de controle, causada pelo aumento dos casos nos anos
anteriores e pelo afrouxamento na aplicação da lei que determina a
erradicação.

O dado exposto na Figura 3 considera o cinturão citrícola em sua totalidade,
entretanto, considerando-se os setores de forma individualizada, dada a
heterogeneidade da disseminação do _greening_ , algumas regiões apresentam
até 60,46% das árvores contaminadas, como em Brotas e 53,18% em Limeira
(FUNDECITRUS, 2020c).

Os fatores que contribuem para a incidência espacialmente heterogênea do
HLB no cinturão estão ligados ao manejo e controle da doença, às
características das propriedades produtoras e ao clima da região. O manejo e
controle da doença estão relacionados, sobretudo, ao esforço e dedicação dos


produtores na erradicação de plantas sintomáticas e no combate ao vetor_._ As
características das unidades produtoras como densidade das fazendas,
tamanho do talhão e idade dos pomares afetam diretamente na ocorrência da
doença, sendo priorizadas as lavouras com árvores mais velhas (mais de 10
anos), de menores proporções (até 10 mil plantas ou 21 hectares) e que estão
densamente distribuídas na região (FUNDECITRUS, 2019).

Além desses fatores, segundo Fundecitrus (2020b), o clima também é definitivo
na disseminação do _greening_ e impacta diretamente os três componentes da
doença: a planta, a bactéria e o psilídeo vetor. Quando da ocorrência de
temperaturas mais altas, associadas principalmente ao déficit hídrico, a taxa de
reprodução da bactéria diminui nas brotações das plantas contaminadas. Em
contrapartida, baixas temperaturas reduzem a quantidade de brotos, principal
atrativo para o inseto, bem como as taxas de reprodução do vetor.

As condições que desfavorecem a disseminação da doença são, portanto, a
baixa população do vetor na região e a menor concentração de bactérias nos
brotos das laranjeiras. Na análise da evolução dos sintomas realizada por
Montesino (2011), constatou-se que o aparecimento e desenvolvimento dos
sintomas estão ligados diretamente às condições de temperatura mais altas.

A detecção do _greening_ pode ser realizada por meio de inspeção visual ou por
métodos analíticos. Os testes analíticos mais comuns utilizados para
diagnóstico e confirmação da doença são microscopia eletrônica, sondas de
DNA, ensaios enzimático e imunoenzimático, Reação em Cadeia da
Polimerase (PCR, do inglês _polymerase chain reaction_ ) e PCR quantitativo
(qPCR) (YAMAMOTO et al., 2006). Desses, o PCR e o qPCR são os teste
laboratoriais mais utilizados e que apresentam maior sensibilidade de detecção
(HANSEN et al., 2008; JUNIOR et al., 2009).

Por outro lado, Deng et al. (2020) afirmam que análises laboratoriais
apresentam condições que podem inviabilizar a detecção, como nível de
complexidade relativamente alto, longo período necessário para o resultado e
demanda de mão de obra qualificada, o que implica em aumento dos custos
necessários para a análise.


Com o método da inspeção visual, os responsáveis pela detecção visam
identificar as plantas infectadas através da observação dos sintomas visíveis,
sendo negligenciadas aquelas árvores que, apesar de infectadas, se mantêm
assintomáticas. Diferentemente dos testes laboratoriais, este é um método
simples e não requer o uso de equipamentos (HALL et al., 2010), entretanto, é
prejudicado pela subjetividade de detecção e requer experiência e
conhecimento específico dos operadores, o que implica na variação da
acurácia do diagnóstico (DENG et al., 2020).

Segundo Junior et al. (2010), a partir do caminhamento a pé, uma equipe de
detecção é capaz de identificar, em média, apenas 47,6% das plantas com
algum sintoma, o que pode ser ainda mais impreciso ao se considerar as
plantas visivelmente assintomáticas. De acordo com Futch, Weingarten e irey
(2009), alterando-se a altura de observação com uso de plataformas, o
intervalo de acurácia desse método pode variar entre 47% e 59%.

Diante disso, é possível afirmar que o desenvolvimento de técnicas de
detecção de HLB que visem praticidade, maior confiabilidade e menor tempo
de resposta é uma demanda urgente da citricultura. Nesse contexto, o uso do
sensoriamento remoto no monitoramento agrícola vem sendo amplamente
estudado nas últimas três décadas (WEISS; JACOB; DUVEILLER, 2020) e tem
significativo potencial para detecção de doenças em lavouras.

As características de incidência e sintomas do _greening_ na agricultura
favorecem o uso do sensoriamento remoto uma vez que corresponde a um
problema de natureza espacial ao se considerar o talhão agrícola e que
provoca alterações na reflectância espectral das árvores (amarelamento das
folhas e ramos, por exemplo), que podem ser identificadas pelos sensores.

Dito isso, será abordado na seção 2.3 o uso do sensoriamento remoto com
foco na gestão e no monitoramento da agricultura, contemplando diferentes
níveis de aquisição e aplicações.


### 2.3. Uso de sensoriamento remoto no monitoramento agrícola

### 2.3.1. Sensoriamento remoto orbital e aéreo

O uso de satélites de observação da terra e de aeronaves tripuladas para
monitoramento da agricultura tem sido pesquisado por mais de 60 anos
(COLWELL, 1956; JACKSON, 1984; PINTER et al., 2003 ) e, segundo Herbei et
al. (2016), têm sido motivado principalmente pelo interesse de trazer mais
eficiência para a atividade.

Os estudos envolvendo essas plataformas fazem uso de imagens de média a
alta resolução espacial e são responsáveis por levantamentos relacionados
principalmente à análise de uso do solo. Por isso, os dados de sensoriamento
remoto apresentam grande importância para agências governamentais no
gerenciamento da agricultura (ALLEN, 1990; DORAISWAMY et al., 2003 ).

Por outro lado, com o uso de imagens satélite as aplicações de escala local,
envolvendo análise dos talhões e das unidades produtivas, não têm sido
exploradas na sua totalidade devido a fatores como resolução espacial
grosseira, tempo de revisita inadequado, presença de nuvens nas imagens e
morosidade na entrega das informações para os usuários, além do custo de
aquisição. (JACKSON, 1984 ; PINTER et al., 2003; MULLA 2013)

Os veículos aéreos não tripulados (VANTs) consistem, no âmbito do
sensoriamento remoto, em uma plataforma do nível aéreo que apresenta
características de obtenção de imagens ideais para a gestão da agricultura na
escala da propriedade produtora. Dentre elas é possível destacar a resolução
espacial refinada, cobertura de aquisição de acordo com a necessidade do
produtor e características da cultura, rápido acesso aos dados gerados e
custos acessíveis (MULLA, 2013; JUNG et al., 2021).

Sendo assim, os VANTs se apresentam como uma boa alternativa para a
identificação e o monitoramento pragas e doenças, sendo um importante aliado
na detecção das áreas de manejo para o controle com defensivos a taxas
variadas (VRA, do inglês Variable Rate Applications) (KRISHNA, 2017), prática
muito aplicada no contexto da agricultura de precisão.


A detecção desses problemas por meio do sensoriamento remoto se dá devido
às alterações das características físicas, químicas e biológicas das plantas
afetadas, o que provoca uma alteração na sua interação com a energia
eletromagnética (alteração da reflectância espectral), identificada pelo sensor
(ZHANG et al., 2019).

Dadrasjavan et al. ( 2019 ) utilizaram um sensor imageador que atua na região
do visível e do infravermelho próximo para detecção de _greening_ em lavouras
de citros. Os autores utilizaram índices de vegetação e as bandas espectrais
com algoritmo SVM (do inglês _Support Vector Machine_ ) para detecção de
plantas e classificação do estado fitossanitário de cada uma delas. A acurácia
atingida na análise foi de 81,75%, demonstrando a eficácia do método
aplicado.

Deng et al. (2020), com uso de sensor acoplado ao VANT e levantamento de
campo com espectrorradiômetro foram capazes de identificar, através de
algoritmos para identificação das melhores regiões do espectro
eletromagnético que indicam a presença da doença, as plantas contaminadas
em plantações de citros com acurácia de classificação superior a 99%.

Garza et al. (2020) analisaram a capacidade de detecção da doença com uso
de VANT equipado com sensor atuante apenas na região do visível, sendo
capaz de obter apenas imagens RGB. O estudo envolveu índices de vegetação
e o algoritmo SVM para classificação. Os autores concluíram que a detecção
do HLB também é possível sem uso de imagens multiespectrais, confirmando o
potencial do uso de VANTs para a atividade.

Utilizando também o algoritmo SVM, Garcia-ruiz et al. (2013) foram capazes de
identificar plantas infectadas pela doença a partir de imagens multiespectrais
captadas com uso de VANT em diferentes alturas e índices de vegetação. As
diferentes alturas proveram imagens com diferentes resoluções espaciais e os
autores concluíram que em ambos os casos, a acurácia de classificação foi
satisfatória, variando de 67 a 85%.


### 2.3.2. Técnicas de processamento digital de imagens

A detecção de doenças em lavouras utilizando imagens de sensoriamento
remoto e técnicas de processamento digital vêm sendo amplamente estudadas
nos últimos anos (MAHLEIN, 2015). Dentre os principais métodos de
classificação de imagens implementados estão os classificadores paramétricos,
que consideram uma distribuição conhecida dos dados e são amplamente
utilizados. As aplicações desses métodos contemplam análises de regressão e
classificações supervisionadas e não supervisionadas.

Entretanto, dados derivados de sensoriamento remoto, assim como os eventos
relacionados ao aparecimento de doenças em copas de árvores, muitas vezes
não apresentam distribuição previamente conhecida, o que inviabiliza a
utilização de métodos paramétricos na análise. Sendo assim, muitos estudos
envolvendo a aprendizagem de máquina ( _machine learning_ ) têm sido
desenvolvidos no âmbito da detecção de doenças em plantações agrícolas
(JUNG et al., 2021).

O aprendizado de máquina é um método empírico capaz de realizar análises
de regressões e classificações a partir de sistemas não lineares e que
apresenta grande potencial no contexto do sensoriamento remoto (LARY et al.,
2016). Além disso, segundo Deng et al. (2020), o rápido desenvolvimento da
computação e da tecnologia de aprendizagem de máquina foi responsável por
um significativo progresso na detecção de HLB.

No contexto do aprendizado de máquinas existe uma abordagem que aplica,
com maior nível de complexidade computacional, os algoritmos baseados em
Redes Neurais Artificiais (RNA) profundas denominado aprendizagem profunda
(DL, do inglês _deep learning_ ) (SCHMIDHUBER, 2015). Os modelos ou redes
de DL são compostos por algumas camadas de funções e pesos aninhadas
entre si que classificam os dados de entrada, como imagens, de acordo com a
análise pretendida por meio da busca e aprendizado progressivo de
transformações dos dados (Figura 4) que os tornem mais discriminantes para a
atividade desejada. (LITJENS et al., 2017, CHOLLET, 2018)


```
Figura 4 – Representação de uma rede neural artificial, evidenciando as camadas e alterações
realizadas nos dados de entrada.
```
Fonte: Chollet (2018).

De acordo com Sujatha et al. (2021), o desempenho das técnicas de _deep
learning_ são superiores àquelas pertencentes ao segmento de _machine
learning_. Os autores realizaram testes de classificação com uso de SVM,
_Random Forest_ (RF) e _Stochastic Gradient Descent_ (SGD) no contexto de ML e
redes neurais convolucionais no âmbito de DL. Todos os testes demonstraram
que o desempenho das abordagens de deep learning são superiores, sendo o
modelo de RF o menos acurado para análise.

As Redes Neurais Convolucionais (CNN, do inglês _convolutional neural
network_ ) são o modelo de DL mais utilizado e foram originalmente
desenvolvidas para processar dados do tipo matricial ou de múltiplos arrays
(LECUN et al. 2015). Devido a essa característica, Ma et al. (2019) pontuam
que as CNNs são bem adaptadas para processar imagens de sensoriamento
remoto, nas quais os dados, ou valores dos pixels e bandas, são arranjados
regularmente em um espaço multidimensional.

De modo prático, as CNNs executam, antes da utilização da rede neural, um
conjunto de filtragens na imagem que são capazes de identificar os melhores
atributos para a classificação desejada. A aplicação dos filtros acontece nas
camadas de convolução e _pooling,_ que correspondem respectivamente à


transformação da imagem destacando padrões de interesse e redução da
resolução espacial (KAMILARIS; PRENAFETA-BOLDÚ, 2018).

A Figura 5 demonstra uma série de imagens que representam as saídas de
cada filtro de uma CNN quando da análise de uma imagem de uma folha
sintomática. É possível observar que ao final de cada processo, elementos
específicos são destacados de modo a revelar mais claramente os sintomas
presentes na folha, principalmente no último passo (Pool5).

```
Figura 5 – Exemplos dos produtos de cada camada de filtros presentes nas CNNs
```
Fonte: Sladojevic et al. (2016).

Zhang et al. (2018) avaliaram a capacidade de identificação de doenças em
folhas de tomateiros a partir do uso de CNNs. Os autores avaliaram três
arquiteturas distintas e atingiram mais de 97% de acurácia global da
classificação. Além disso, os autores concluíram que a análise realizada tem
grande potencial de ser generalizada e expandida para outras doenças
vegetais.

Zeng et al. (2020) realizaram uma análise comparativa com modelos de _deep
learning_ envolvendo diversas arquiteturas de CNN para avaliação da
capacidade de detecção do HLB em folhas de citros com base em uma
metodologia de aumento de amostras. Os autores relatam que o uso da técnica
aplicada no aumento das amostras de treinamento ocasionou uma melhora
significativa da análise.

Além da identificação de doenças em plantas, as CNNs são utilizadas em
diversas aplicações no âmbito da visão computacional, como classificação de


uso e cobertura da terra, classificação de imagens no geral e detecção de
objetos (MA et al., 2019). Uma das aplicações que será abordada nesta
pesquisa é o reconhecimento e delimitação de objetos presentes nas imagens
de sensoriamento remoto.

Braga et al. (2020) realizaram um estudo de identificação e delimitação de
copas de árvores em florestas tropicais com uso de CNNs. Os autores
adotaram a arquitetura Mask R-CNN aplicada a imagens de altíssima resolução
espacial e atingiram resultados promissores, com aproximadamente 60 mil
árvores detectadas e delineadas com até 91% de precisão. A arquitetura de
rede convolucional Mask R-CNN também foi adotada em Braga et al. (2019)
para delimitação de copas de árvores em imagens de sensoriamento remoto
com resultados promissores.


## 3. MATERIAIS E MÉTODOS

### 3.1. Área de estudo

A área de estudo dessa pesquisa (Figura 6 ) é composta por frações de dois
talhões comerciais localizados no cinturão citrícola. Os talhões pertencem a
fazendas situadas na porção paulista do cinturão e ambos estão na região de
Limeira, no setor Sul. O talhão A está localizado no município de Cosmópolis, a
aproximadamente 9,2 km da sede municipal e a 5,3 km da cidade de
Holambra, o talhão B localiza-se no município de Mogi Mirim, distante 5,9 km
da sede.

```
Figura 6 – Localização da área de estudo
```
Fonte: Elaborado pelo autor.

Os talhões não serão totalmente contemplados, de modo que seja viável a
inspeção visual de todas as laranjeiras em um único dia. Por isso, foram
selecionadas duas frações de cada talhão, resultando numa área total de 5,02
hectares para o talhão A e 4,86 para o talhão B.

Os critérios de escolha da região de Limeira se deram pelo grau de
disseminação da doença no setor, que ocupa a segunda posição entre as
regiões mais afetadas do cinturão, e pela proximidade da cidade de São José
dos Campos, de modo que sejam reduzidos os custos com deslocamento para


trabalhos de campo. As fazendas e talhões foram selecionados de acordo com
a aceitação dos respectivos produtores quanto ao levantamento dos dados nas
propriedades.

### 3.2. Dados

### 3.2.1. Imagens VANT

As imagens utilizadas nesta pesquisa serão adquiridas a partir do sobrevoo de
um VANT equipado com um sensor multiespectral. As definições de modelos
dos dois equipamentos ainda serão definidas. Contudo, as bandas espectrais
de atuação do sensor imageador deverão contemplar tanto a região do visível
quanto o infravermelho próximo do espectro eletromagnético, sendo possível a
geração de todos os índices de vegetação previstos (comentados na subseção
3.3.2).

O levantamento deverá ocorrer entre os horários de 1 1 e 14 horas em dias nos
quais as condições atmosféricas estejam favoráveis para o imageamento, isto
é, com poucas nuvens ou com céu limpo. A data do levantamento será definida
no decorrer da pesquisa, entretanto, espera-se que sejam realizadas duas
campanhas, previstas para os meses de maio e junho de 2021.

As imagens serão obtidas sempre a nadir, de modo que a integridade
geométrica das cenas seja mantida. As imagens serão levantadas com
sobreposição lateral e longitudinal de 70 a 80% e a altura de voo será definida
conforme a necessidade da resolução espacial. Para correção geométrica e
radiométrica das imagens, serão utilizados pontos de controles ocupados com
receptores GNSS e placas de calibração radiométrica.

### 3.2.2. Inspeção visual

A inspeção visual ocorrerá no mesmo dia do levantamento das imagens e será
realizada por uma equipe de trabalho previamente treinada e instruída. As
árvores serão rotuladas e classificadas pelos agentes da inspeção de modo
que seja viável a montagem do banco de dados geográfico.


### 3.3. Metodologia

As etapas de processamentos sugeridos neste estudo são apresentadas na
Figura 7 e a descrição de cada procedimento é detalhada nas subseções
adiante.

```
Figura 7 – Fluxograma da metodologia de trabalho proposta
```
Fonte: Elaborado pelo autor.

### 3.3.1. Pré-processamento

O pré-processamento é primeiro passo realizado após o imageamento e
consiste na conversão das cenas em formato JPG em imagens geolocalizadas
no formato TIF e na aplicação das correções radiométrica e geométrica, que
tem como produtos o ortomosaico de reflectância e o Modelo Digital de


Superfície (MDS) da área imageada (DADRASJAVAN et al., 2019; LAN et al.,
2020).

O processo de correção radiométrica será realizado a partir do cômputo do
fator de reflectância (Fr) espectral, que representa uma medida indireta da
reflectância dos alvos presentes na cena. Para esse procedimento, painéis de
controle calibrados e com características conhecidas serão utilizados durante o
voo e os dados coletados subsidiarão o cálculo a partir da seguinte equação
(CLEMENS, 2012):

Onde corresponde ao fator de reflectância do alvo, e são

o valor do pixel observado na imagem no instante e é o fator de
reflectância do painel, que é fornecido pelo fabricante do equipamento.

A correção geométrica se dará pelo registro das bandas espectrais e pela
geração dos ortomosaicos das duas áreas. Segundo Dadrasjavan et al. (2019),
o deslocamento entre as bandas causado pelo imageamento de baixa altitude
não pode ser negligenciado, uma vez que gera grandes anomalias no resultado
da álgebra de mapa, por exemplo, etapa responsável pela geração dos índices
de vegetação.

O registro das bandas e a geração do ortomosaico serão realizados com o uso
de algoritmo de detecção de feições, como o SIFT (do inglês, _scale_ ‐ _invariant
feature transform_ ) (LOWE, 1999). A partir da sobreposição espacial entre as
bandas de uma mesma imagem e entre imagens vizinhas, o algoritmo é capaz
de identificar as feições presentes em mais de uma cena e realizar o
alinhamento entre as bandas e a geração do mosaico espacialmente acurado.
Ainda, para assegurar a qualidade espacial no processo de ortorretificação do
mosaico, serão utilizados os pontos de controle levantados em campo com
receptor GNSS.

Além do ortomosaico, o pré-processamento das imagens irá gerar o MDS da
área através de técnicas de fotogrametria. O método SfM (do inglês, _Structure
from Motion_ ) gera uma nuvem tridimensional de pontos que foram imageados


por mais de um ângulo de observação e detectados por algoritmos como o
SIFT. A visualização dos pontos por diferentes ângulos é possível devido à
sobreposição longitudinal e lateral entre as imagens. Por meio da
generalização da nuvem de pontos será gerado o MDS das duas cenas.

As etapas do pré-processamento das imagens serão realizadas em dois
ambientes e os resultados gerados serão comparados entre si. O primeiro
corresponde ao software comercial Pix4Dmapper (Pix4D SA, Suíça) e o
segundo, à biblioteca python de código aberto denominada OpenDroneMap
(ODM). A ferramenta ODM fornece duas possibilidades de processamento ao
usuário, abrangendo tanto a interface de interação quanto a execução de
processos por linha de comando.

O uso da ferramenta ODM por linha de comando permitirá avaliar a viabilidade
de implementação de um algoritmo automatizado que contempla todos os
passos envolvidos na análise pretendida por esse trabalho, que corresponde
aos procedimentos de pré-processamento, manipulação dos dados, análise e
validação.

Além do pré-processamento das imagens, será realizado o pré-processamento
dos dados levantados em campo. Nessa etapa, os dados coletados serão
digitalizados e ocorrerá a seleção das amostras de treinamento e teste. A priori,
as amostras serão divididas em 80/20, isto é, 80% das árvores presentes nos
talhões serão utilizadas para treinamento do algoritmo e 20% para teste.

### 3.3.2. Extração de atributos

Além das composições de imagens geradas pelas diferentes bandas espectrais
obtidas pelo sensor, será avaliado o desempenho da identificação de plantas
infectadas a partir da composição de imagens formadas por principais
componentes, como realizado em Bargoti e Underwood (2017), e com por
diferentes índices de vegetação (IV), de acordo com Lan et al. (2020).

É interessante notar que a extração de atributos não é um procedimento
necessário quando do uso de redes neurais profundas, uma vez que o modelo
é capaz de extrair as próprias representações que melhor discriminam o evento
estudado. Entretanto, serão adotadas novas composições além daquelas


geradas com as bandas espectrais, de acordo com os estudos mencionados no
parágrafo anterior, para comparação dos resultados.

Segundo Mirsha et al. (2011) e (2009), os índices de vegetação sugeridos no
campo de identificação do HLB são NDVI, SIPI, TVI, DVI, RVI, SR, G, MCARI1,
MTVI-1, MTVI-2 e RDVI. Desses, serão utilizados os cinco primeiros e os seus
detalhes são expostos na Tabela 1.

```
Tabela 1 – Índices de vegetação adotados
Sigla Nome Equação Fonte
NDVI Normalised Difference Vegetation Index Peñuelas et al., 1997
```
```
SIPI Structure Intensive Pigment^ Peñuelas et al., 1995
```
```
TVI Triangular Vegetation Index^ Haboudane et al., 2004
DVI Difference Vegetation Index Becker et al., 1988
```
RVI _Ratio Vegetation Index_ (^) Jordan et al., 1969
Fonte: Elaborado pelo autor.
A análise de principais componentes (PCA, do inglês _principal components
analysis_ ) é definida matematicamente com uma transformação linear ortogonal
que projeta o conjunto de dados em um novo sistema de referência. A
transformação possibilita a formação de novos eixos de coordenadas, ou
principais componentes (PC), que definem as variáveis transformadas
(CENTENO, 2009). Os produtos da PCA são novas imagens (ou bandas) de
representação do espaço sem redundância de informação que serão usadas
para gerar composições de imagens.
Os procedimentos de extração de atributos serão realizados com uso da
linguagem de programação Python e, caso o uso dos IVs e PCs resultem em
melhores resultados, eles serão considerados na avaliação da viabilidade do
sistema de monitoramento.

### 3.3.3. Segmentação

O estudo a ser realizado nesta pesquisa considera como unidade de análise
cada árvore presente no talhão comercial, que é representada na imagem pelo
seu dossel. Diante disso, será realizada a identificação dos dosséis presentes


no ortomosaico a partir da segmentação de instância de objetos. De modo
geral, a segmentação de instância de objetos objetiva delinear os contornos
dos elementos presentes na cena a partir da detecção dos alvos de interesse
por retângulo envolvente e posterior identificação dos pixels que o compõem
(DOLL; GIRSHICK; AI, 2017).

A arquitetura _Mask_ R-CNN será utilizada no processo de identificação das
copas individualizadas. Para isso, a rede será treinada com um conjunto de
amostras geradas a partir das imagens obtidas. A geração de amostras será
realizada com o uso dos ortomosaicos e dos MDS para delimitação manual de
árvores presentes no talhão e terá como produto um conjunto de feições
vetoriais (polígonos) que envolvem os pixels pertencentes a cada planta usada
como amostra.

Para auxiliar no processo de delimitação manual das copas, os ortomosaicos
serão filtrados por um limiar aplicado ao MDS de modo que sejam
desconsiderados os pixels que representam a parcela de solo, que
corresponde à região entre linhas de plantio e entre árvores (quando houver).
Desse modo, o MDS irá proporcionar melhor discriminação das árvores quando
avaliadas as regiões de borda das copas confrontantes com regiões de solo,
sobretudo quando essas apresentarem alguma cobertura vegetal.

O levantamento das amostras de treinamento do modelo irá considerar as duas
áreas imageadas, assim, as amostras dos dois talhões não serão segregadas e
farão parte do mesmo conjunto de dados de treinamento. Uma vez levantadas
as amostras, a _Mask_ R-CNN será submetida ao processo de treinamento e seu
desempenho será avaliado.

De posse do modelo treinado, os dois ortomosaicos serão submetidos à etapa
de segmentação de instância dos objetos e espera-se que sejam delimitadas
todas as árvores presentes nas imagens. Após a segmentação, será realizada
a validação do resultado, como sugerido por Braga et al. (2020), com a geração
de pontos aleatórios na região da vegetação – filtrada inicialmente pelo MDS – ,
avaliação visual da pertinência da classificação de cada um deles e geração da
matriz de confusão.


### 3.3.4. Classificação

Após a identificação de cada árvore presente nos talhões imageados, será
realizado o processo de classificação em relação à contaminação e severidade
da doença. A classificação será executada com o uso de uma segunda CNN e
o seu treinamento será realizado a partir das amostras selecionadas no pré-
processamento. A arquitetura da rede de classificação será definida de acordo
com a viabilidade de implementação.

No processo de geração das amostras (Figura 8 ), serão utilizadas as
composições de imagens formadas com as bandas espectrais, índices de
vegetação e principais componentes, além da delimitação de cada copa
produzida no passo da segmentação. Todas as composições serão recortadas
nos limites de cada polígono considerando um aumento horizontal e vertical de
10% da sua extensão. Após o recorte, as amostras serão rotuladas de acordo
com o levantamento realizado em campo.

```
Figura 8 – Fluxograma da etapa de classificação
```
Fonte: Elaborado pelo autor.

A quantidade de amostras necessárias para treinamento de CNNs é um fator
determinante para a acurácia final da classificação (MA et al., 2019). Dito isso,
será adotado o procedimento de aumento das amostras a partir da
manipulação espacial de cada uma delas. Segundo Zeng et al. (2020), os
procedimentos de aumento de amostras de treinamento são realizados, entre
outros, pelo espelhamento horizontal e vertical, rotação e mudança de escala
( _zoom_ ). Assim, para cada amostra de treinamento serão produzidas quatro
novas amostras, definidas pelas transformações mencionadas.


Esse procedimento pode viabilizar o treinamento do modelo com os diferentes
níveis de severidade da doença, uma vez que, com a divisão em quatro
estágios de contaminação, a quantidade de amostras de cada grupo pode ficar
reduzida e impossibilitar a classificação de amostras diferentes daquelas
usadas no treinamento (evento indesejado conhecido como _overfitting_ ). Os
níveis de severidade adotados serão 1, 2, 3 e 4, de acordo com a classificação
realizada pelo Fundecitrus.

Além disso, para avaliação da capacidade de monitoramento da doença e
diagnósticos em áreas distintas, a rede será treinada e avaliada com uso de
diferentes combinações de amostras de treinamento e teste. Será avaliada a
acurácia da classificação treinando e avaliando o modelo com amostras de
cada área individualmente (treinamento e teste apenas na área A e apenas na
área B); treinamento com amostras de uma área e teste na outra; e
treinamento com amostras das duas áreas simultaneamente.

De posse do modelo treinado, as amostras serão submetidas à classificação de
acordo com as combinações de amostras mencionadas e posteriormente será
realizada a validação do resultado. A validação será realizada com base no
levantamento de campo e medidas de acurácia serão utilizadas para expor o
resultado. Diante da possibilidade de equívoco no levantamento de campo, as
árvores classificadas erroneamente na análise poderão ser submetidas ao
teste de PCR para confirmação da sua condição fitossanitária.

A implementação das CNNs utilizadas na segmentação e na classificação será
realizada em Python com uso de bibliotecas de construção e manipulação de
modelos de ML e SL, como o TensorFlow.


## 4. RESULTADOS ESPERADOS

Os resultados esperados nessa pesquisa contemplam os quatros objetivos
específicos, a saber:

```
 Delimitação de todas as copas das árvores presentes nos talhões
imageados com um nível de acurácia adequado, de acordo com os
resultados encontrados na literatura. Esse resultado, além de necessário
para o desenvolvimento do estudo, também será útil para os produtores
devido à possibilidade da contagem automatizada de árvores na
propriedade.
 Determinação, a partir do resultado da classificação, dos melhores
atributos que indicam a contaminação da doença. Esse resultado
indicará a necessidade ou não do uso de atributos, além das bandas
espectrais geradas pelo imageamento, em análises futuras.
 Classificação de cada árvore identificada de acordo com seu estado
fitossanitário e, em casos de contaminação pelo HLB, identificação do
nível de severidade da doença. Caso seja possível a identificação das
árvores infectadas em estágios iniciais, esse resultado será de extrema
importância para o procedimento de erradicação.
 Avaliação, após a implementação de todos os processamentos, da
viabilidade de criação de um sistema de monitoramento que recebe
como entrada as imagens de talhões agrícolas de laranja e, a partir de
um modelo previamente treinado para identificação e classificação das
plantas, tem como saída o estado fitossanitário de cada uma delas. É
importante notar que a variabilidade dos fatores que envolvem o estudo,
como as características de imageamento (horário, mês, altura, sensor,
dentre outros), diferentes espécies de laranjas e idade dos pomares
aumentam a complexidade da análise e demanda um conjunto de
amostras que contempla todas essas variações e, por isso, esse
resultado esperado consiste apenas em uma avaliação preliminar.
```

## 5. CRONOGRAMA

Etapas (^) ABR MAI JUN JUL AGO^2021 SET OUT NOV DEZ JAN FEV^2022 MAR
Pesquisa bibliográfica X X X X X X X X X X X
Apresentação da proposta X
Aquisição das imagens X X
Pré-processamento X X X
Extração dos atributos X
Geração das amostras de treinamento da
segmentação X^ X^
CNN de segmentação
Treinamento X
Teste X X
Validação X
Geração das amostras de classificação X
CNN de classificação
Treinamento X
Teste X X
Validação X
Adequação da proposta com as sugestões da banca X X
Elaboração e submissão do artigo X X X X X X X X X
Redação da dissertação X X X X X X X X X X
Defesa da dissertação X


#### 6. REFERÊNCIAS BIBLIOGRÁFICAS

ALLEN, J. D. A Look at the Remote Sensing Applications Program of the
National Agricultural Statistics Service. **Journal of Official Statistics** v. 6 p.
393. 1990.

BASSANEZI, R. B.; MONTESINO, L. H.; BUSATO, L. A.; STUCHI, E. S.
Damages caused by huanglongbing on sweet orange yield and quality in São
Paulo. **Proceedings of the Huanglongbing - Greening International
Workshop**. Ribeirão Preto SP. p. 39. 2006

BARGOTI, S.; UNDERWOOD, J. Deep fruit detection in orchards. **Proceedings**

**- IEEE International Conference on Robotics and Automation** , p. 3626–
3633, 2017.

BASSANEZI, R. B.; LOPES, S. A.; DE MIRANDA, M. P.; WULFF, N. A.;
VOLPE, H. X. L.; AYRES, A. J. Overview of citrus huanglongbing spread and
management strategies in Brazil. **Tropical Plant Pathology** , v. 45, n. 3, p. 251–
264, 2020.

BOTEON, M.; NEVES, E. M. Citricultura brasileira: aspectos econômicos. In:
**Citros.** Campinas - SP: Instituto Agronômico, Fundag, 2005. p. 20–36.

BRAGA, J. R. G.; PERIPATO, V.; DALAGNOL, R.; FERREIRA, M. P.;
TARABALKA, Y.; ARAGÃO, L. E. O. C.; DE CAMPOS VELHO, H. F.;
SHIGUEMORI, E. H.; WAGNER, F. H. Tree crown delineation algorithm based
on a convolutional neural network. **Remote Sensing** , v. 12, n. 8, p. 1–27, 2020.

BRAGA, J. R. G.; VELHO, H. F. C.; SHIGUEMORI, E. H.; WAGNER, F. H.
Algoritmo de delineação de copas baseado em deep learning. **Anais do XIX
Simpósio Brasileiro de Sensoriamento Remoto** , p. 136 6 – 1369, 2019.

CENTENO, J. A. S. **Sensoriamento remoto e Processamento de Imagens
Orbitais**. Curitiba: Curso de Pós Graduação em Ciências Geodésicas, UFPR.
Curitiba, PR. 2009.

CHOLLET, F. **Deep learning with python**. [s.l: s.n.]. ISBN(9780996452762).

CLEMENS, S. R. **Procedures for Correcting Digital Camera Imagery**


**Acquired by the AggieAir Remote Sensing Platform**. 2012. UTAH STATE
UNIVERSITY, 2012.

DA GRAÇA, J. V. Citrus greening disease. **Annual review of phytopathology.
Vol. 29** , n. 192, p. 109–136, 1991.

DADRASJAVAN, F.; SAMADZADEGAN, F.; SEYED POURAZAR, S. H.;
FAZELI, H. **UAV-based multispectral imagery for fast Citrus Greening
detection**. **Journal of Plant Diseases and Protection** 2019.

DENG, X.; ZHU, Z.; YANG, J.; ZHENG, Z.; HUANG, Z.; YIN, X.; WEI, S.; LAN,
Y. Detection of citrus huanglongbing based on multi-input neural network model
of UAV hyperspectral remote sensing. **Remote Sensing** , v. 12, n. 17, p. 1–20,
2020.

DOLL, P.; GIRSHICK, R.; AI, F. Mask R-CNN ar. **Proceedings of the IEEE
International Conference on Computer Vision (ICCV)** , p. 2961–2969, 2017.

ERPEN, L.; MUNIZ, F. R.; MORAES, T. DE S.; TAVANO, E. C. DA R. Análise
do cultivo da laranja no Estado de São Paulo de 2001 a 2015. **Revista
IPecege** , v. 4, n. 1, p. 33–43, 2018.

FAO - Food and Agriculture Organization of the United Nations. Crops. 2019.
Disponível em: <http://www.fao.org/faostat/en/#data/QC>. Acesso em: 10 mar.
2021.

FRANCO, A. S. M. **O Suco de laranja brasileiro no mercado global**. **Análise
Conjuntural** 2016. Disponível em:
<http://www.ipardes.gov.br/biblioteca/docs/bol_38_6_c.pdf>.

FUNDECITRUS. Levantamento da incidência das doenças dos citros:
Greening, CVC e cancro cítrico. **Fundecitrus** , 2019.

FUNDECITRUS. **Estimativa da safra de laranja 2020/21 São Paulo e
Triângulo/Sudoeste Mineiro**. 2020a. Disponível em:
<http://marefateadyan.nashriyat.ir/node/150>

FUNDECITRUS. **Inventário de árvores do cinturão citrícola de São Paulo e
Triângulo/Sudoeste Mineiro: retrato dos pomares em março de 2020**.


**Fundecitrus** 2020b.

FUNDECITRUS. **Levantamento da incidência das doenças dos citros:
Greening, CVC e cancro cítrico no Cinturão Citrícola de São Paulo e
Triângulo/Sudoeste Mineiro 2019**. **Fundecitrus** Araraquara, SP: 2020c.

FUTCH, S.; WEINGARTEN, S.; IREY, M. Determining HLB Infection Levels
using Multiple Survey Methods in Florida Citrus. **Procedings of the Florida
State Horticultural Society** , v. 122, p. 152–157, 2009.

GARCIA-RUIZ, F.; SANKARAN, S.; MAJA, J. M.; LEE, W. S.; RASMUSSEN, J.;
EHSANI, R. Comparison of two aerial imaging platforms for identification of
Huanglongbing-infected citrus trees. **Computers and Electronics in
Agriculture** , v. 91, p. 106–115, 2013.

GARZA, B. N.; ANCONA, V.; ENCISO, J.; PEROTTO-BALDIVIESO, H. L.;
KUNTA, M.; SIMPSON, C. Quantifying citrus tree health using true color UAV
images. **Remote Sensing** , v. 12, n. 1, p. 1–13, 2020.

HALL, D. G.; SHATTERS, R. G.; CARPENTER, J. E.; SHAPIRO, J. P.
Research toward an artificial diet for adult Asian citrus psyllid. **Annals of the
Entomological Society of America** , v. 103, n. 4, p. 611–617, 2010.

HERBEI, M. V.; POPESCU, C. A.; BERTICI, R.; SMULEAC, A.; POPESCU, G.
Processing and Use of Satellite Images in Order to Extract Useful Information in
Precision Agriculture. **Bulletin of University of Agricultural Sciences and
Veterinary Medicine Cluj-Napoca. Agriculture** , v. 73, n. 2, p. 238, 2016.

IBGE - Instituto Brasileiro de Geografia e Estatística. Produção Agrícola
Municipal – 2019. Disponível em:
<https://sidra.ibge.gov.br/pesquisa/pam/tabelas>. Acesso em: 10 mar. 20 21.

JR, E. R. H.; DAUGHTRY, C. S. T. What good are unmanned aircraft systems
for agricultural remote sensing and precision agriculture ? **International
Journal of Remote Sensing** , v. 39, n. 15–16, p. 5345–5376, 2018.

JUNG, J.; MAEDA, M.; CHANG, A.; BHANDARI, M.; ASHAPURE, A.;
LANDIVAR-BOWLES, J. The potential of remote sensing and artificial


intelligence as tools to improve the resilience of agriculture production systems.
**Current Opinion in Biotechnology** , v. 70, p. 15–22, 2021.

JUNIOR, J. B.; FILHO, A. B.; BASSANEZI, R. B.; BARBOSA, J. C.;
FERNANDES, N. G.; YAMAMOTO, P. T.; LOPES, S. A.; MACHADO, M. A.;
JUNIOR, P. L.; AYRES, A. J.; MASSARI, C. A. Base científica para a
erradicação de plantas sintomáticas e assintomáticas de huanglongbing (HLB,
greening) visando o controle efetivo da doença. **Tropical Plant Pathology** , v.
34, n. 3, p. 137–145, 2009.

JUNIOR, J. B.; YAMAMOTO, P. T.; MIRANDA, M. P. DE; BASSANEZI, R. B.;
AYRES, A. J.; BOVÉ, J. M. Controle do huanglongbing no estado de São
Paulo, Brasil. **Citrus Research & Technology** , v. 31, n. 1, p. 53–64, 2010.

KAMILARIS, A.; PRENAFETA-BOLDÚ, F. X. Deep learning in agriculture: A
survey. **Computers and Electronics in Agriculture** , v. 147, n. February, p.
70 – 90, 2018.

LAN, Y.; HUANG, Z.; DENG, X.; ZHU, Z.; HUANG, H.; ZHENG, Z.; LIAN, B.;
ZENG, G.; TONG, Z. Comparison of machine learning methods for citrus
greening detection on UAV multispectral images. **Computers and Electronics
in Agriculture** , v. 171, n. January, 2020.

LARY, D. J.; ALAVI, A. H.; GANDOMI, A. H.; WALKER, A. L. Machine learning
in geosciences and remote sensing. **Geoscience Frontiers** , v. 7, n. 1, p. 3–10,
2016.

LIN, H.; CHEN, C.; DODDAPANENI, H.; DUAN, Y.; CIVEROLO, E. L.; BAI, X.;
ZHAO, X. A new diagnostic system for ultra-sensitive and specific detection and
quantification of Candidatus Liberibacter asiaticus, the bacterium associated
with citrus Huanglongbing. **Journal of Microbiological Methods** , v. 81, n. 1, p.
17 – 25, 2010.

MA, L.; LIU, Y.; ZHANG, X.; YE, Y.; YIN, G.; JOHNSON, B. A. Deep learning in
remote sensing applications: A meta-analysis and review. **ISPRS Journal of
Photogrammetry and Remote Sensing** , v. 152, n. March, p. 166–177, 2019.

MONTESINO, L. H. **Evolução dos sintomas de Huanglongbing em**


**laranjeiras jovens: relação com época do ano, fenologia das plantas,
flutuação populacional de Diaphorina citri Kuwayama (Hemiptera:
Psyllidae) e medidas de controle do vetor**. 2011. 46 p. Fundo de Defesa da
Citricultura, 2011.

NETO, G. G. **Perfil e tendências da cultura da laranja dentro do cinturão
citrícola (São Paulo e triângulo/sudoeste mineiro) para o citricultor**. 2019.
Universidade Estadual Paulista, 2019.

OSCO, L. P.; NOGUEIRA, K.; MARQUES RAMOS, A. P.; FAITA PINHEIRO, M.
M.; FURUYA, D. E. G.; GONÇALVES, W. N.; DE CASTRO JORGE, L. A.;
MARCATO JUNIOR, J.; DOS SANTOS, J. A. Semantic segmentation of citrus-
orchard using deep neural networks and multispectral UAV-based imagery.
**Precision Agriculture** , n. 0123456789, 2021.

PAULO, S. **Citricultura gera mais de 48 mil vagas de empregos no Estado
em 2019**. Disponível em:
<https://www.saopaulo.sp.gov.br/spnoticias/citricultura-gera-mais-de- 48 - mil-
vagas-de-empregos-no-estado-em-2019/>. Acesso em: 15 mar. 2020.

RODRIGUES, J. D. B.; MOREIRA, A. S.; STUCHI, E. S.; BASSANEZI, R. B.;
LARANJEIRA, F. F.; GIRARDI, E. A. Huanglongbing incidence, canopy volume,
and sprouting dynamics of ‘Valencia’ sweet orange grafted onto 16 rootstocks.
**Tropical Plant Pathology** , v. 45, n. 6, p. 611–619, 2020.

ROSSI, F. R. **Determinantes da adoção de irrigação por citricultores da
região centro-norte do Estado de São Paulo**. 2017. 254 p. Universidade
Federal de São Carlos, 2017.

SUJATHA, R.; CHATTERJEE, J. M.; JHANJHI, N. Z.; BROHI, S. N.
Performance of deep learning vs machine learning in plant leaf disease
detection. **Microprocessors and Microsystems** , v. 80, n. November 2020, p.
103615, 2021.

WEISS, M.; JACOB, F.; DUVEILLER, G. Remote sensing for agricultural
applications: A meta-review. **Remote Sensing of Environment** , v. 236, n.
August 2019, p. 111402, 2020.


ZENG, Q.; MA, X.; CHENG, B.; ZHOU, E.; PANG, W. GANs-Based Data
Augmentation for Citrus Disease Severity Detection Using Deep Learning. **IEEE
Access** , v. 8, p. 172882–172891, 2020.

ZHANG, K.; WU, Q.; LIU, A.; MENG, X. Can deep learning identify tomato leaf
disease? **Advances in Multimedia** , v. 2018, 2018.



