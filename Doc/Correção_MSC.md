# O que é MSC (Multiplicative Signal Correction)?

O **MSC** é uma técnica de **pré-processamento de sinais** usada principalmente em espectroscopia (como **NIR – infravermelho próximo**) para **corrigir variações indesejadas** causadas por efeitos físicos como:

- Diferenças de **espalhamento de luz**,
- **Desvios de caminho óptico**,
- **Diferenças na amostra** (ex: compactação, umidade).

Essas variações **não estão relacionadas à composição química da amostra**, mas afetam os espectros. Por isso, é importante corrigi-las **antes de usar modelos preditivos** como PLS, PCA, entre outros.

---

## Intuição

Imagine que você tenha vários espectros (linhas com valores) e cada um está **"esticado" ou "amplificado"** de forma diferente em relação a um espectro ideal (referência).  

O MSC corrige cada espectro para que ele **fique mais parecido com esse espectro de referência**, ajustando uma **reta**:

\[
x_{\text{corrigido}} = \frac{x - b}{a}
\]

Onde:

- \( x \): espectro original  
- \( a \): inclinação (correção multiplicativa)  
- \( b \): intercepto (correção aditiva)  
- \( x_{\text{corrigido}} \): espectro corrigido

---

## Como se calcula?

Para cada espectro \( x \):

1. Escolha um **espectro de referência** (normalmente a **média dos espectros**).
2. Ajuste uma **regressão linear simples** entre o espectro \( x \) e o espectro de referência \( x_{\text{ref}} \):

\[
x = a \cdot x_{\text{ref}} + b
\]

3. Use os coeficientes para corrigir o espectro:

\[
x_{\text{corrigido}} = \frac{x - b}{a}
\]

Assim, os espectros ficam **normalizados** e prontos para análises mais confiáveis.

---
