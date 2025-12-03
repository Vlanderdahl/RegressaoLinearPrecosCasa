using CSV
using DataFrames
using Statistics
using Plots
using JSON



println("=" ^ 60)
println("Iniciando análise exploratória de dados...")
println("=" ^ 60)

#Carregar o dataframe limpo
println("\n[1/7] Carregando dataset limpo...")
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
println("  Colunas: $(names(df))")

#Descrever os dados
println("\n[2/7] Descrevendo os dados...")
println("\n" * "=" ^ 60)
println("Estatísticas descritivas:")
println("=" ^ 60)
println(describe(df))

#Calcular correlações
println("\n[3/7] Calculando correlações entre variáveis...")

#Selecionar apenas colunas numéricas para correlação
colunas_numericas = []
for col in names(df)
    if eltype(df[!, col]) <: Union{Int64, Float64}
        push!(colunas_numericas, col)
    end
end

println("  Colunas numéricas para correlação: $colunas_numericas")

#Criar matriz de dados numéricos
dados_numericos = Matrix{Float64}(df[!, colunas_numericas])

#Calcular matriz de correlação
matriz_correlacao = cor(dados_numericos)

println("\n" * "=" ^ 60)
println("Matriz de correlação:")
println("=" ^ 60)

#Criar DataFrame com a matriz de correlação para melhor visualização
df_correlacao = DataFrame(matriz_correlacao, [Symbol(col) for col in colunas_numericas])
println(df_correlacao)

#Focar em correlações com preco_de_venda
println("\n[4/7] Correlações com a variável alvo 'preco_de_venda':")
println("=" ^ 60)

if "preco_de_venda" in colunas_numericas
    idx_preco = findfirst(x -> x == "preco_de_venda", colunas_numericas)
    
    println("\nCorrelações com preco_de_venda (ordenadas por valor absoluto):")
    correlacoes_preco = []
    
    for (i, col) in enumerate(colunas_numericas)
        if col != "preco_de_venda"
            corr_valor = matriz_correlacao[i, idx_preco]
            push!(correlacoes_preco, (col, corr_valor))
        end
    end
    
    #Ordenar por valor absoluto da correlação
    sort!(correlacoes_preco, by = x -> abs(x[2]), rev = true)
    
    for (col, corr) in correlacoes_preco
        simbolo = corr > 0.5 ? "🔺" : corr < -0.5 ? "🔻" : "➖"
        println("  $simbolo $col: $(round(corr, digits=4))")
    end
    
    #Preparar dados de correlação para exportação JSON
    global correlacoes_dict = Dict(
        "variavel_alvo" => "preco_de_venda",
        "correlacoes" => Dict(col => corr for (col, corr) in correlacoes_preco),
        "matriz_completa" => Dict(
            "colunas" => colunas_numericas,
            "valores" => matriz_correlacao
        )
    )
else
    println("⚠ Variável 'preco_de_venda' não encontrada nas colunas numéricas")
    global correlacoes_dict = Dict()
end

#Gerar histograma da variável alvo
println("\n[5/7] Gerando histograma de preco_de_venda...")

if "preco_de_venda" in names(df)
    try
        histograma = histogram(
            df.preco_de_venda,
            bins = 50,
            xlabel = "Preço de Venda",
            ylabel = "Frequência",
            title = "Distribuição de Preços de Venda",
            legend = false,
            color = :steelblue,
            grid = true,
            size = (800, 600)
        )
        
        savefig(histograma, "histograma_preco.png")
        println("✓ Histograma salvo em: histograma_preco.png")
    catch e
        println("⚠ Erro ao gerar histograma: $e")
    end
else
    println("⚠ Variável 'preco_de_venda' não encontrada")
end

#Análises adicionais (top valores)
println("\n[6/7] Realizando análises adicionais...")
println("=" ^ 60)

#Top preços por quantidade_banheiros
if "quantidade_banheiros" in names(df) && "preco_de_venda" in names(df)
    println("\n📊 Preço médio por quantidade de banheiros:")
    preco_por_banheiros = combine(
        groupby(df, "quantidade_banheiros"),
        "preco_de_venda" => mean => "preco_medio",
        "preco_de_venda" => std => "preco_desvio",
        nrow => "quantidade"
    )
    sort!(preco_por_banheiros, "preco_medio", rev = true)
    println(preco_por_banheiros)
    
    #Adicionar ao dicionário de resultados
    if !haskey(correlacoes_dict, "analises_adicionais")
        correlacoes_dict["analises_adicionais"] = Dict()
    end
    correlacoes_dict["analises_adicionais"]["preco_por_banheiros"] = Dict(
        row.quantidade_banheiros => Dict(
            "preco_medio" => row.preco_medio,
            "preco_desvio" => row.preco_desvio,
            "quantidade" => row.quantidade
        ) for row in eachrow(preco_por_banheiros)
    )
end

#Top preços por existe_segundo_andar
if "existe_segundo_andar" in names(df) && "preco_de_venda" in names(df)
    println("\n📊 Preço médio por existência de segundo andar:")
    preco_por_andar = combine(
        groupby(df, "existe_segundo_andar"),
        "preco_de_venda" => mean => "preco_medio",
        "preco_de_venda" => std => "preco_desvio",
        nrow => "quantidade"
    )
    println(preco_por_andar)
    
    if !haskey(correlacoes_dict, "analises_adicionais")
        correlacoes_dict["analises_adicionais"] = Dict()
    end
    correlacoes_dict["analises_adicionais"]["preco_por_segundo_andar"] = Dict(
        row.existe_segundo_andar => Dict(
            "preco_medio" => row.preco_medio,
            "preco_desvio" => row.preco_desvio,
            "quantidade" => row.quantidade
        ) for row in eachrow(preco_por_andar)
    )
end

#Top preços por qualidade_da_cozinha_Excelente
if "qualidade_da_cozinha_Excelente" in names(df) && "preco_de_venda" in names(df)
    println("\n📊 Preço médio por qualidade da cozinha (Excelente):")
    preco_por_cozinha = combine(
        groupby(df, "qualidade_da_cozinha_Excelente"),
        "preco_de_venda" => mean => "preco_medio",
        "preco_de_venda" => std => "preco_desvio",
        nrow => "quantidade"
    )
    println(preco_por_cozinha)
    
    if !haskey(correlacoes_dict, "analises_adicionais")
        correlacoes_dict["analises_adicionais"] = Dict()
    end
    correlacoes_dict["analises_adicionais"]["preco_por_cozinha_excelente"] = Dict(
        row.qualidade_da_cozinha_Excelente => Dict(
            "preco_medio" => row.preco_medio,
            "preco_desvio" => row.preco_desvio,
            "quantidade" => row.quantidade
        ) for row in eachrow(preco_por_cozinha)
    )
end

#Estatísticas gerais
if "preco_de_venda" in names(df)
    println("\n📊 Estatísticas gerais de preco_de_venda:")
    println("  Média: $(round(mean(df.preco_de_venda), digits=2))")
    println("  Mediana: $(round(median(df.preco_de_venda), digits=2))")
    println("  Desvio padrão: $(round(std(df.preco_de_venda), digits=2))")
    println("  Mínimo: $(round(minimum(df.preco_de_venda), digits=2))")
    println("  Máximo: $(round(maximum(df.preco_de_venda), digits=2))")
    println("  Quantidade: $(nrow(df))")
    
    if !haskey(correlacoes_dict, "estatisticas_gerais")
        correlacoes_dict["estatisticas_gerais"] = Dict()
    end
    correlacoes_dict["estatisticas_gerais"] = Dict(
        "media" => mean(df.preco_de_venda),
        "mediana" => median(df.preco_de_venda),
        "desvio_padrao" => std(df.preco_de_venda),
        "minimo" => minimum(df.preco_de_venda),
        "maximo" => maximum(df.preco_de_venda),
        "quantidade" => nrow(df)
    )
end

#Exportar resultados de correlação para JSON
println("\n[7/7] Exportando resultados de correlação para JSON...")

arquivo_json = "correlacoes_resultados.json"
try
    open(arquivo_json, "w") do f
        JSON.print(f, correlacoes_dict, 4)
    end
    println("✓ Resultados exportados para: $arquivo_json")
catch e
    println("✗ Erro ao exportar JSON: $e")
    rethrow(e)
end

#Resumo final
println("\n" * "=" ^ 60)
println("Resumo da análise:")
println("=" ^ 60)
println("✓ Dataset carregado: $(nrow(df)) linhas × $(ncol(df)) colunas")
println("✓ Matriz de correlação calculada: $(length(colunas_numericas)) variáveis numéricas")
println("✓ Histograma gerado: histograma_preco.png")
println("✓ Análises adicionais realizadas")
println("✓ Resultados exportados: $arquivo_json")

println("\n" * "=" ^ 60)
println("Análise exploratória concluída com sucesso!")
println("=" ^ 60)

