using CSV
using DataFrames
using Statistics



println("=" ^ 60)
println("Iniciando ingestão e limpeza de dados...")
println("=" ^ 60)

#Carregar o CSV
println("\n[1/9] Carregando arquivo CSV...")
try
    global df = CSV.read("Preços_de_casas.csv", DataFrame)
    println("✓ Arquivo carregado com sucesso!")
    println("  Dimensões iniciais: $(nrow(df)) linhas × $(ncol(df)) colunas")
catch e
    println("✗ Erro ao carregar CSV: $e")
    rethrow(e)
end

#Imprimir amostras para depuração
println("\n[2/9] Exibindo amostras dos dados...")
println("\nPrimeiras 5 linhas:")
println(first(df, 5))
println("\nÚltimas 5 linhas:")
println(last(df, 5))
println("\nInformações do DataFrame:")
println(describe(df))
println("\nNomes das colunas:")
println(names(df))

#Remover duplicatas
println("\n[3/9] Removendo duplicatas...")
n_antes = nrow(df)
df = unique(df)
n_depois = nrow(df)
duplicatas_removidas = n_antes - n_depois
println("✓ Duplicatas removidas: $duplicatas_removidas")
println("  Dimensões após remoção: $(nrow(df)) linhas × $(ncol(df)) colunas")

#Tratar valores ausentes
println("\n[4/9] Tratando valores ausentes...")
println("Valores ausentes por coluna:")
for col in names(df)
    n_missing = count(ismissing, df[!, col])
    if n_missing > 0
        println("  $col: $n_missing valores ausentes")
    end
end

#Remover linhas com valores ausentes em colunas chave
colunas_chave = ["preco_de_venda"]
println("\nRemovendo linhas com valores ausentes em: $(colunas_chave)")
n_antes = nrow(df)
df = dropmissing(df, colunas_chave)
n_depois = nrow(df)
missing_removidos = n_antes - n_depois
println("✓ Linhas removidas por valores ausentes: $missing_removidos")
println("  Dimensões após tratamento: $(nrow(df)) linhas × $(ncol(df)) colunas")

#Filtrar preco_de_venda > 0
println("\n[5/9] Filtrando preços de venda > 0...")
n_antes = nrow(df)
df = filter(row -> row.preco_de_venda > 0, df)
n_depois = nrow(df)
precos_invalidos = n_antes - n_depois
println("✓ Linhas removidas (preço ≤ 0): $precos_invalidos")
println("  Dimensões após filtro: $(nrow(df)) linhas × $(ncol(df)) colunas")

#Converter capacidade_carros_garagem de ft² para m² (dividir por 10.764)
println("\n[6/9] Convertendo capacidade_carros_garagem de ft² para m²...")
if "capacidade_carros_garagem" in names(df)
    df.capacidade_carros_garagem = df.capacidade_carros_garagem ./ 10.764
    println("✓ Conversão realizada (dividido por 10.764)")
    println("  Exemplo de valores convertidos:")
    println("  Primeiros 5 valores: $(first(df.capacidade_carros_garagem, 5))")
else
    println("⚠ Coluna capacidade_carros_garagem não encontrada")
end

#Identificar e remover outliers usando z-score
println("\n[7/9] Identificando e removendo outliers em preco_de_venda usando z-score...")
if "preco_de_venda" in names(df)
    precos = df.preco_de_venda
    media = mean(precos)
    desvio = std(precos)
    
    println("  Média: $media")
    println("  Desvio padrão: $desvio")
    
    #Calcular z-scores
    z_scores = (precos .- media) ./ desvio
    
    #Identificar outliers (|z-score| > 3)
    outliers = abs.(z_scores) .> 3
    n_outliers = sum(outliers)
    
    println("  Outliers identificados (|z-score| > 3): $n_outliers")
    
    if n_outliers > 0
        println("  Exemplos de outliers:")
        outliers_df = df[outliers, :]
        println(first(outliers_df[!, ["Id", "preco_de_venda"]], min(5, n_outliers)))
    end
    
    #Remover outliers
    n_antes = nrow(df)
    df = df[.!outliers, :]
    n_depois = nrow(df)
    println("✓ Outliers removidos: $n_outliers")
    println("  Dimensões após remoção: $(nrow(df)) linhas × $(ncol(df)) colunas")
else
    println("⚠ Coluna preco_de_venda não encontrada")
end

#Converter tipos de colunas para apropriados
println("\n[8/9] Convertendo tipos de colunas...")

#Áreas em Float64
colunas_areas = ["area_primeiro_andar", "area_segundo_andar", "capacidade_carros_garagem"]
for col in colunas_areas
    if col in names(df)
        df[!, col] = convert.(Float64, df[!, col])
        println("  ✓ $col convertido para Float64")
    end
end

#Quantidades em Int64
colunas_quantidades = ["Id", "existe_segundo_andar", "quantidade_banheiros", "qualidade_da_cozinha_Excelente"]
for col in colunas_quantidades
    if col in names(df)
        df[!, col] = convert.(Int64, df[!, col])
        println("  ✓ $col convertido para Int64")
    end
end

#Preço em Float64
if "preco_de_venda" in names(df)
    df.preco_de_venda = convert.(Float64, df.preco_de_venda)
    println("  ✓ preco_de_venda convertido para Float64")
end

println("\nTipos finais das colunas:")
println(eltype.(eachcol(df)))

#Exportar para Parquet com fallback
println("\n[9/9] Exportando dados limpos...")
arquivo_saida_parquet = "precos_limpo.parquet"
arquivo_saida_csv = "precos_limpo.csv"

global parquet_sucesso = false
try
    using Parquet
    try
        Parquet.writefile(arquivo_saida_parquet, df)
        global parquet_sucesso = true
    catch
        Parquet.write(arquivo_saida_parquet, df)
        global parquet_sucesso = true
    end
catch
    try
        using Arrow
        Arrow.write(arquivo_saida_parquet, df)
        global parquet_sucesso = true
    catch e
        println("⚠ Parquet/Arrow não disponíveis: $e")
    end
end

if parquet_sucesso
    println("✓ Dados exportados para: $arquivo_saida_parquet")
    println("  Dimensões finais: $(nrow(df)) linhas × $(ncol(df)) colunas")
else
    println("⚠ Exportando para CSV (Parquet não disponível ou falhou)...")
    try
        CSV.write(arquivo_saida_csv, df)
        println("✓ Dados exportados para: $arquivo_saida_csv")
        println("  Dimensões finais: $(nrow(df)) linhas × $(ncol(df)) colunas")
    catch e2
        println("✗ Erro ao exportar para CSV: $e2")
        rethrow(e2)
    end
end

#Validações finais
println("\n" * "=" ^ 60)
println("Validações finais:")
println("=" ^ 60)
println("✓ Número de linhas: $(nrow(df))")
println("✓ Número de colunas: $(ncol(df))")
println("✓ Valores ausentes em preco_de_venda: $(count(ismissing, df.preco_de_venda))")
println("✓ Preços mínimos: $(minimum(df.preco_de_venda))")
println("✓ Preços máximos: $(maximum(df.preco_de_venda))")
println("✓ Preços médios: $(round(mean(df.preco_de_venda), digits=2))")

println("\n" * "=" ^ 60)
println("Processo de ingestão e limpeza concluído com sucesso!")
println("=" ^ 60)
