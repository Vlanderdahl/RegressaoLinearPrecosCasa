using CSV
using DataFrames
using GLM
using Statistics
using Random



println("=" ^ 60)
println("Treinando modelo de regressão linear...")
println("=" ^ 60)

#Configurar seed para reprodutibilidade
Random.seed!(42)

#Carregar o dataframe limpo
println("\n[1/8] Carregando dataset limpo...")
arquivo_parquet = "precos_limpo.parquet"
arquivo_csv = "precos_limpo.csv"

global df = nothing
try
    try
        using Parquet2
        global df = Parquet2.readfile(arquivo_parquet)
        println("✓ Dataset carregado de: $arquivo_parquet")
    catch
        try
            using Parquet
            global df = Parquet.readfile(arquivo_parquet)
            println("✓ Dataset carregado de: $arquivo_parquet")
        catch
            global df = CSV.read(arquivo_csv, DataFrame)
            println("✓ Dataset carregado de: $arquivo_csv (fallback)")
        end
    end
catch e
    println("✗ Erro ao carregar dataset: $e")
    rethrow(e)
end

println("  Dimensões: $(nrow(df)) linhas × $(ncol(df)) colunas")

#Selecionar features e alvo
println("\n[2/8] Selecionando features e variável alvo...")

features = [
    "area_primeiro_andar",
    "existe_segundo_andar",
    "area_segundo_andar",
    "quantidade_banheiros",
    "capacidade_carros_garagem",
    "qualidade_da_cozinha_Excelente"
]

target = "preco_de_venda"

#Verificar se todas as colunas existem
colunas_faltantes = []
for col in [features; target]
    if !(col in names(df))
        push!(colunas_faltantes, col)
    end
end

if !isempty(colunas_faltantes)
    println("✗ Colunas faltantes: $colunas_faltantes")
    error("Colunas necessárias não encontradas no dataset")
end

println("✓ Features selecionadas: $features")
println("✓ Variável alvo: $target")

#Criar DataFrame apenas com features e target
colunas_modelo = [features; target]
df_modelo = df[!, colunas_modelo]

println("  Dimensões do dataset para modelagem: $(nrow(df_modelo)) linhas × $(ncol(df_modelo)) colunas")

#Verificar valores ausentes
println("\n[3/8] Verificando valores ausentes...")
for col in colunas_modelo
    n_missing = count(ismissing, df_modelo[!, col])
    if n_missing > 0
        println("  ⚠ $col: $n_missing valores ausentes")
    end
end

#Remover valores ausentes se houver
n_antes = nrow(df_modelo)
df_modelo = dropmissing(df_modelo)
n_depois = nrow(df_modelo)
if n_antes != n_depois
    println("  ✓ Removidas $(n_antes - n_depois) linhas com valores ausentes")
end

println("  Dimensões finais: $(nrow(df_modelo)) linhas")

#Dividir em train/test (80/20)
println("\n[4/8] Dividindo dataset em treino (80%) e teste (20%)...")

n_total = nrow(df_modelo)
n_train = Int(floor(0.8 * n_total))
n_test = n_total - n_train

#Embaralhar índices
indices = shuffle(1:n_total)
indices_train = indices[1:n_train]
indices_test = indices[(n_train+1):end]

train_df = df_modelo[indices_train, :]
test_df = df_modelo[indices_test, :]

println("  ✓ Dataset de treino: $n_train linhas ($(round(100*n_train/n_total, digits=1))%)")
println("  ✓ Dataset de teste: $n_test linhas ($(round(100*n_test/n_total, digits=1))%)")

#Construir fórmula do modelo
println("\n[5/8] Construindo fórmula do modelo...")

# Criar fórmula: preco_de_venda ~ feature1 + feature2 + ...
formula_str = "$target ~ " * join(features, " + ")
formula_modelo = eval(Meta.parse("(@formula($formula_str))"))

println("  Fórmula: $formula_modelo")

#Treinar o modelo
println("\n[6/8] Treinando modelo de regressão linear...")

try
    global modelo = lm(formula_modelo, train_df)
    println("✓ Modelo treinado com sucesso!")
catch e
    println("✗ Erro ao treinar modelo: $e")
    rethrow(e)
end

#Exibir resumo do modelo
println("\n" * "=" ^ 60)
println("Resumo do modelo:")
println("=" ^ 60)
println(modelo)

#Fazer predições no test set
println("\n[7/8] Fazendo predições no conjunto de teste...")

try
    global predicoes = predict(modelo, test_df)
    println("✓ Predições realizadas para $(length(predicoes)) observações")
catch e
    println("✗ Erro ao fazer predições: $e")
    rethrow(e)
end

#Calcular métricas de avaliação
valores_reais = test_df[!, target]
rmse = sqrt(mean((predicoes .- valores_reais).^2))
mae = mean(abs.(predicoes .- valores_reais))
r2 = cor(predicoes, valores_reais)^2

println("\n" * "=" ^ 60)
println("Métricas de avaliação no conjunto de teste:")
println("=" ^ 60)
println("  RMSE (Root Mean Squared Error): $(round(rmse, digits=2))")
println("  MAE (Mean Absolute Error): $(round(mae, digits=2))")
println("  R² (Coeficiente de determinação): $(round(r2, digits=4))")
println("  Média dos valores reais: $(round(mean(valores_reais), digits=2))")
println("  Média das predições: $(round(mean(predicoes), digits=2))")

#Extrair e imprimir coeficientes
println("\n[8/8] Analisando coeficientes do modelo...")
println("\n" * "=" ^ 60)
println("Coeficientes do modelo e interpretação:")
println("=" ^ 60)

coeficientes = coef(modelo)
nomes_coef = coefnames(modelo)

for (nome, coef_valor) in zip(nomes_coef, coeficientes)
    println("\n📊 $nome: $(round(coef_valor, digits=2))")
    
    #Interpretação específica para cada variável
    if nome == "(Intercept)"
        println("   → Preço base estimado quando todas as features são zero.")
    elseif nome == "area_primeiro_andar"
        println("   → Cada m² adicional no primeiro andar aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
    elseif nome == "existe_segundo_andar"
        if coef_valor > 0
            println("   → Ter um segundo andar (vs. não ter) aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
        else
            println("   → Ter um segundo andar (vs. não ter) diminui o preço em aproximadamente R\$ $(round(abs(coef_valor), digits=2)).")
            println("   ⚠ Nota: Este coeficiente negativo pode indicar multicolinearidade com area_segundo_andar.")
        end
    elseif nome == "area_segundo_andar"
        println("   → Cada m² adicional no segundo andar aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
    elseif nome == "quantidade_banheiros"
        println("   → Cada banheiro adicional aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
    elseif nome == "capacidade_carros_garagem"
        println("   → Cada m² adicional na capacidade da garagem aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
    elseif nome == "qualidade_da_cozinha_Excelente"
        println("   → Ter cozinha de qualidade Excelente (vs. não ter) aumenta o preço em aproximadamente R\$ $(round(coef_valor, digits=2)).")
    else
        println("   → Impacto no preço: $(round(coef_valor, digits=2)) por unidade.")
    end
end

#Salvar modelo ou coeficientes
println("\n[9/9] Salvando modelo e coeficientes...")

#Salvar coeficientes em CSV
arquivo_coeficientes = "coeficientes_modelo.csv"
df_coeficientes = DataFrame(
    variavel = nomes_coef,
    coeficiente = coeficientes
)
CSV.write(arquivo_coeficientes, df_coeficientes)
println("✓ Coeficientes salvos em: $arquivo_coeficientes")

#Salvar métricas em arquivo de texto
arquivo_metricas = "metricas_modelo.txt"
open(arquivo_metricas, "w") do f
    write(f, "Métricas do Modelo de Regressão Linear\n")
    write(f, "=" ^ 50 * "\n\n")
    write(f, "RMSE: $(round(rmse, digits=2))\n")
    write(f, "MAE: $(round(mae, digits=2))\n")
    write(f, "R²: $(round(r2, digits=4))\n\n")
    write(f, "Coeficientes:\n")
    for (nome, coef_valor) in zip(nomes_coef, coeficientes)
        write(f, "$nome: $(round(coef_valor, digits=2))\n")
    end
end
println("✓ Métricas salvas em: $arquivo_metricas")

#Tentar salvar o modelo completo (se possível)
try
    eval(:(using JLD2))
    arquivo_modelo = "modelo_regressao.jld2"
    eval(:(@save $arquivo_modelo $modelo))
    println("✓ Modelo completo salvo em: $arquivo_modelo")
catch e
    println("⚠ JLD2 não disponível. Modelo completo não foi salvo (apenas coeficientes).")
    println("  Para salvar o modelo completo, instale JLD2: using Pkg; Pkg.add(\"JLD2\")")
end

#Resumo final
println("\n" * "=" ^ 60)
println("Resumo do treinamento:")
println("=" ^ 60)
println("✓ Dataset carregado: $(nrow(df_modelo)) observações")
println("✓ Features utilizadas: $(length(features))")
println("✓ Tamanho do treino: $n_train observações")
println("✓ Tamanho do teste: $n_test observações")
println("✓ RMSE no teste: $(round(rmse, digits=2))")
println("✓ R² no teste: $(round(r2, digits=4))")
println("✓ Coeficientes salvos: $arquivo_coeficientes")

println("\n" * "=" ^ 60)
println("Treinamento do modelo concluído com sucesso!")
println("=" ^ 60)

