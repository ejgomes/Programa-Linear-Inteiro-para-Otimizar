# Programação-Linear-Inteira-para-Otimização-na-Produção-Sunícola
Esse repositório apresenta um modelo de programação linear inteira com PuLP para otimizar a produção de carne suína. O objetivo é maximizar o lucro, considerando demanda, disponibilidade e rendimento dos cortes. Resultados mostram aumento das margens e estratégias eficazes. Futuras melhorias incluirão controle de estoque.

#!pip install pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import pandas as pd
import matplotlib.pyplot as plt

# Dados do problema
produtos = ["Porco Inteiro", "Metade Porco", "Kit Feijoada", "Pernil", "Lombo", "Kit Geral"]
custos_producao = [4.50, 3.00, 5.00, 7.50, 8.00, 5.00]  # Custo de produção fixo
precos_venda = [12.50, 6.50, 25.00, 20.50, 24.00, 11.50]  # Preço de venda por kg
demanda_min = [150, 220, 120, 150, 130, 100]  # Demanda mínima semanal (em kg)
demanda_max = [300, 400, 230, 280, 240, 200]  # Demanda máxima semanal (em kg)
porcos_disponiveis_base = 500  # Porcos disponíveis semanalmente
rendimentos = [1, 2, 1, 2, 2, 2]  # Quantidade de produto gerada por porco (em kg)

# Modelo de otimização
modelo = LpProblem("Maximizacao_Lucro_Frigorifico", LpMaximize)

# Variáveis de decisão (kg produzidos para cada produto)
x = LpVariable.dicts("Quantidade", produtos, lowBound=0, cat="Integer")

# Variáveis de porcos usados para diferentes combinações
y_metade = LpVariable("Porcos_Para_Metade", lowBound=0, cat="Integer")
y_lombo = LpVariable("Porcos_Para_Lombo", lowBound=0, cat="Integer")
y_pernil = LpVariable("Porcos_Para_Pernil", lowBound=0, cat="Integer")
y_kits = LpVariable("Porcos_Para_Kits", lowBound=0, cat="Integer")

# Função objetivo: Maximizar o lucro líquido
lucro_liquido = lpSum((precos_venda[i] - custos_producao[i]) * x[produtos[i]] for i in range(len(produtos)))
modelo += lucro_liquido, "Lucro_Total"

# Restrições de produção (ligadas ao número de porcos disponíveis)
modelo += y_metade + y_lombo + y_pernil + y_kits <= porcos_disponiveis_base, "Restricao_Porcos_Disponiveis"

# Restrições para ligar produção aos porcos usados
modelo += x["Metade Porco"] == rendimentos[1] * y_metade, "Restricao_Metade"
modelo += x["Lombo"] == rendimentos[4] * y_lombo, "Restricao_Lombo"
modelo += x["Pernil"] == rendimentos[3] * y_pernil, "Restricao_Pernil"
modelo += x["Kit Feijoada"] + x["Kit Geral"] == rendimentos[5] * y_kits, "Restricao_Kits"

# Subprodutos gerados junto com lombo, pernil, ou kits
modelo += x["Pernil"] >= x["Lombo"], "Subproduto_Pernil_Lombo"
modelo += x["Kit Feijoada"] >= x["Lombo"], "Subproduto_KitFeijoada_Lombo"
modelo += x["Kit Geral"] >= x["Lombo"], "Subproduto_KitGeral_Lombo"

# Restrições de demanda mínima e máxima
for i in range(len(produtos)):
    modelo += x[produtos[i]] >= demanda_min[i], f"Demanda_Minima_{produtos[i]}"
    modelo += x[produtos[i]] <= demanda_max[i], f"Demanda_Maxima_{produtos[i]}"

# Função para resolver o modelo com diferentes números de porcos
def resolver_modelo(porcos_disponiveis):
    modelo.constraints["Restricao_Porcos_Disponiveis"].changeRHS(porcos_disponiveis)
    modelo.solve()
    lucro = modelo.objective.value()
    return lucro, {p: x[p].value() for p in produtos}

# Testar diferentes variações de porcos disponíveis
variacoes = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
lucros = []
quantidades = []
porcos_testados = []

for v in variacoes:
    porcos_variados = int(porcos_disponiveis_base * (1 + v))
    porcos_testados.append(porcos_variados)
    lucro, qtd = resolver_modelo(porcos_variados)
    lucros.append(lucro)
    quantidades.append(qtd)

# Resultados em tabela
df_resultados = pd.DataFrame({"Porcos Disponíveis": porcos_testados, "Lucro Total": lucros})
print(df_resultados)

# Exibir detalhes de quantidades produzidas
print("\nDetalhes das quantidades produzidas por variação:")
for i, qtd in enumerate(quantidades):
    print(f"\nPorcos Disponíveis: {porcos_testados[i]}")
    for produto, quantidade in qtd.items():
        print(f"{produto}: {quantidade} kg")

# Gráfico: Lucro Total x Porcos Disponíveis
plt.figure(figsize=(8, 6))
plt.plot(df_resultados["Porcos Disponíveis"], df_resultados["Lucro Total"], marker="o")
plt.title("Lucro Total em função do Número de Porcos Disponíveis")
plt.xlabel("Porcos Disponíveis")
plt.ylabel("Lucro Total")
plt.grid()
plt.show()
