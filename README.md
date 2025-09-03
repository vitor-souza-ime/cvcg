# CVCG - Benchmark de Processamento de Imagens com NAO e PyTorch

Repositório: [https://github.com/vitor-souza-ime/cvcg](https://github.com/vitor-souza-ime/cvcg)

---

## Descrição

Este projeto realiza um **benchmark completo de filtros de imagens** utilizando a câmera do robô NAO, comparando implementações na **CPU** (OpenCV) e **GPU** (PyTorch). O objetivo é avaliar o **speedup** obtido ao utilizar GPU em diferentes operações, resoluções e tamanhos de batch.

O benchmark inclui filtros clássicos como **Gaussian**, **Sobel**, **FFT high-pass**, além de versões em lote (batch) para GPU com PyTorch, aproveitando **convoluções 2D** otimizadas.

Resultados são salvos em gráficos e arquivos CSV para análise posterior.

---

## Estrutura do Repositório

```

cvcg/
│
├─ main.py                  # Código principal do benchmark
├─ benchmark\_results.csv    # Resultados gerados (após execução)
├─ comprehensive\_benchmark\_results.png # Gráficos gerados
└─ README.md                # Documentação do projeto

````

---

## Requisitos

- Python 3.10+  
- NAOqi SDK (para conectar à câmera do NAO)  
- OpenCV (`opencv-python`)  
- NumPy  
- PyTorch com suporte a CUDA (GPU)  
- Matplotlib  
- Pandas  
- SciPy  

Instalação sugerida via pip:

```bash
pip install opencv-python numpy torch matplotlib pandas scipy
````

> Certifique-se de que o PyTorch esteja instalado com suporte à sua GPU:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Uso

1. Conecte o robô NAO à mesma rede que a máquina onde o script será executado.
2. Configure o IP do NAO no `main.py`:

```python
ip = "172.15.1.80"  # Substitua pelo IP do seu NAO
port = 9559
```

3. Execute o benchmark:

```bash
python main.py
```

4. Durante a execução, o script irá capturar frames, executar filtros na CPU e GPU, medir tempos, calcular speedup e gerar gráficos.

5. Os resultados serão salvos em:

* `benchmark_results.csv` → tabela completa de tempos e speedups
* `comprehensive_benchmark_results.png` → gráficos comparativos

---

## Funcionalidades

* Captura de frames da câmera do NAO
* Redimensionamento para diferentes resoluções
* Benchmark de filtros clássicos (CPU) e otimizados (GPU):

  * Gaussian
  * Sobel
  * FFT High-pass
* Processamento em lote (batch) para GPU
* Análise de speedup por batch size e resolução
* Visualização de resultados em gráficos
* Exportação de dados em CSV

---

## Contribuição

Contribuições são bem-vindas!

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Faça commit das suas alterações (`git commit -m 'Adicionar nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

