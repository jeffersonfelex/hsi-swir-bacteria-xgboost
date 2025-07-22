# Transformada Logarítmica (Log(1/R))

## O que é?

A **Transformada Logarítmica** é usada para transformar valores de **reflectância (R)** em **absorvância (A)**, realçando **bandas de baixa intensidade** e comprimindo **altas amplitudes**:

\[
A = \log\left(\frac{1}{R}\right) = -\log(R)
\]

## Quando aplicar?

- Quando bandas importantes têm **baixa intensidade**;
- Para **linearizar relações não lineares**;
- Quando os dados espectrais apresentam **grande variação de amplitude**.

## Intuição

- Picos altos de reflectância são comprimidos;
- Pequenos detalhes nos espectros se tornam mais visíveis;
- Mais adequado para análise estatística (PLS, PCA, etc).

