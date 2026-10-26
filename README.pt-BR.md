# AIMED: classificação de imagens de ECG

**1º lugar na tarefa de classificação do George B. Moody PhysioNet Challenge 2024.**

[English](README.md) | [Artigo](https://doi.org/10.22489/CinC.2024.398) | [Pesos](https://huggingface.co/fmenegui/cinc_of5)

O modelo recebe a imagem de uma página de ECG e estima 11 possíveis achados.
Usamos uma ConvNeXt-Tiny e combinamos os cinco modelos dos folds pela média
das probabilidades. A classificação é feita diretamente na imagem.

Este repositório adapta o código de `fmenegui/cinc_of5`, commit
`00da6ef97f52f475da86895c286538d3fa8f7c4b`, para facilitar instalação,
treinamento e uso. As alterações estão documentadas. O macro F1 de **0,817** é o
resultado histórico do Challenge. Não representa uma nova avaliação desta versão.

## Instalar

Use Python 3.10 a 3.12. Execute os comandos na raiz do repositório.

```bash
git clone https://github.com/fmenegui/aimed_physionet2024.git
cd aimed_physionet2024
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Escolha uma das opções de PyTorch:

```bash
# CPU
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

```bash
# GPU NVIDIA, pacotes CUDA 11.8
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

Instale as demais dependências:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e media
```

No Windows, a ativação é `.venv\Scripts\activate`. Comece com
`--num-workers 0` no treinamento. A automação de testes usa Linux.

## Usar o ensemble pronto

```bash
python download_weights.py --kind ensemble
python predict_image.py -i /caminho/ecg.png -m model -o outputs/predicao.json
```

O download dos cinco modelos tem aproximadamente 557 MB. O script fixa a revisão
dos pesos e verifica a integridade de cada arquivo. Não exige Git LFS nem token
do Hugging Face. Para usar modelos já treinados, o checkpoint de pré-treinamento
não é necessário.

O resultado JSON contém os escores das 11 classes, os rótulos selecionados com
limiar `0.5` e os arquivos de pesos utilizados. Um ECG pode receber vários rótulos.
Os escores não foram calibrados como probabilidades clínicas.

Para verificar a instalação com desenhos artificiais:

```bash
python examples/make_demo_data.py --output data/demo --records 20
python predict_image.py -i data/demo/demo000-0.png -m model
```

Os desenhos e os rótulos do exemplo servem apenas para testar o programa.
Não são exames de pacientes nem um conjunto de avaliação clínica.

## Treinar

Organize os dados com imagens e cabeçalhos `.hea` que indiquem `# Image:` e
`# Labels:`. O [tutorial de dados](docs/data.md) explica a preparação do PTB-XL,
a inclusão dos rótulos com PTB-XL+ e a geração de imagens com ECG-Image-Kit.

```bash
python download_weights.py --kind pretrained
python train.py -d data/training -m runs/experimento01
```

Isso treina os cinco folds a partir do checkpoint de ECG publicado. A configuração
padrão usa imagens de 384 × 384, BCE ponderada, AdamW, lotes de 16, até 30 épocas
e parada antecipada após cinco épocas sem melhora suficiente da perda de validação.

Para começar com um único fold e um lote menor:

```bash
python train.py -d data/training -m runs/fold0 --fold 0 --batch-size 4
```

Para testar um passo de treinamento em CPU sem baixar pesos:

```bash
python train.py -d data/demo -m runs/teste-demo --init none --fold 0 --epochs 1 --batch-size 2 --size 64 --num-workers 0 --device cpu --fast-dev-run --quiet
```

Esse teste verifica o funcionamento do código. Seus pesos e métricas não têm
significado diagnóstico. Os modelos finais de um treinamento ficam em
`runs/experimento01/model/`, prontos para uso com `predict_image.py`.

## O que é possível reproduzir

| Etapa | Disponibilidade |
| --- | --- |
| Inferência com os cinco modelos publicados | Pesos e código disponíveis |
| Ajuste final no PTB-XL usando o checkpoint de ECG | Código e checkpoint disponíveis |
| Treinamento alternativo a partir de ImageNet ou pesos aleatórios | Opções `--init imagenet` e `--init none` |
| Recriar todo o pré-treinamento original | Depende dos dados privados InCor-AMB e InCor-EMG |
| Repetir a avaliação oficial de 0,817 | Depende do conjunto oculto do Challenge |

As opções alternativas de inicialização não reproduzem o treinamento vencedor.
A divisão padrão mantém as imagens do mesmo registro no mesmo fold. Ela não
garante separação por paciente quando há vários ECGs da mesma pessoa.

## Guias completos

- [Preparar os dados](docs/data.md)
- [Treinar, escolher parâmetros e avaliar](docs/training.md)
- [Usar em Python ou no formato do Challenge](docs/inference.md)
- [Método, diferenças da submissão e limites de reprodução](docs/reproducibility.md)
- [Repositórios de outras equipes usados como referência de documentação](docs/references.md)
- [Testes executados nesta versão](docs/validation.md)
- [Situação das licenças](LICENSING.md)

Os guias detalhados estão em inglês para facilitar o uso pela comunidade do
Challenge. Para citar o trabalho, consulte [CITATION.cff](CITATION.cff).
