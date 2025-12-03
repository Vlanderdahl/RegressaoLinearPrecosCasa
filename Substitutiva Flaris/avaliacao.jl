using CSV
using DataFrames
using GLM
using Statistics
using Random
using Plots



println("=" ^ 60)
println("Avaliando modelo de regressão linear...")
println("=" ^ 60)

#Configurar seed para reprodutibilidade
Random.seed!(42)

#Carregar dataset limpo
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

#Preparar dados (mesma divisão train/test do modelo_regressao.jl)
println("\n[2/7] Preparando dados para avaliação...")

features = [
    "area_primeiro_andar",
    "existe_segundo_andar",
    "area_segundo_andar",
    "quantidade_banheiros",
    "capacidade_carros_garagem",
    "qualidade_da_cozinha_Excelente"
]

target = "preco_de_venda"

#Criar DataFrame apenas com features e target
colunas_modelo = [features; target]
df_modelo = df[!, colunas_modelo]
df_modelo = dropmissing(df_modelo)

#Dividir em train/test
n_total = nrow(df_modelo)
n_train = Int(floor(0.8 * n_total))
n_test = n_total - n_train

indices = shuffle(MersenneTwister(42), 1:n_total)
indices_train = indices[1:n_train]
indices_test = indices[(n_train+1):end]

train_df = df_modelo[indices_train, :]
test_df = df_modelo[indices_test, :]

println("  ✓ Dataset de teste: $n_test observações")

#Carregar ou treinar modelo
println("\n[3/7] Carregando ou treinando modelo...")

global modelo = nothing
try
    try
        eval(:(using JLD2))
        eval(:(@load "modelo_regressao.jld2" modelo))
        println("✓ Modelo carregado de: modelo_regressao.jld2")
    catch
        formula_str = "$target ~ " * join(features, " + ")
        formula_modelo = eval(Meta.parse("(@formula($formula_str))"))
        global modelo = lm(formula_modelo, train_df)
        println("✓ Modelo treinado novamente")
    end
catch e
    println("✗ Erro ao carregar/treinar modelo: $e")
    rethrow(e)
end

#Fazer predições no test set
println("\n[4/7] Fazendo predições no conjunto de teste...")

try
    global predicoes = predict(modelo, test_df)
    global valores_reais = test_df[!, target]
    println("✓ Predições realizadas para $(length(predicoes)) observações")
catch e
    println("✗ Erro ao fazer predições: $e")
    rethrow(e)
end

#Calcular métricas
println("\n[5/7] Calculando métricas de avaliação...")

#MSE
mse = mean((predicoes .- valores_reais).^2)

#RMSE
rmse = sqrt(mse)

#R² (Coeficiente de determinação)
#R² = 1 - (SS_res / SS_tot)
ss_res = sum((valores_reais .- predicoes).^2)
ss_tot = sum((valores_reais .- mean(valores_reais)).^2)
r2 = 1 - (ss_res / ss_tot)

#MAE
mae = mean(abs.(predicoes .- valores_reais))

#R² alternativo usando correlação
r2_cor = cor(predicoes, valores_reais)^2

println("\n" * "=" ^ 60)
println("Métricas de avaliação:")
println("=" ^ 60)
println("  MSE (Mean Squared Error): $(round(mse, digits=2))")
println("  RMSE (Root Mean Squared Error): $(round(rmse, digits=2))")
println("  MAE (Mean Absolute Error): $(round(mae, digits=2))")
println("  R² (Coeficiente de determinação): $(round(r2, digits=4))")
println("  R² (via correlação): $(round(r2_cor, digits=4))")
println("  Média dos valores reais: $(round(mean(valores_reais), digits=2))")
println("  Média das predições: $(round(mean(predicoes), digits=2))")
println("  Desvio padrão dos valores reais: $(round(std(valores_reais), digits=2))")
println("  Desvio padrão das predições: $(round(std(predicoes), digits=2))")

#Gerar scatter plot de valores reais vs preditos
println("\n[6/7] Gerando visualizações...")

try
    #Valores reais vs preditos
    scatter_plot = scatter(
        valores_reais,
        predicoes,
        xlabel = "Valores Reais (R\$)",
        ylabel = "Valores Preditos (R\$)",
        title = "Valores Reais vs Preditos\nR² = $(round(r2, digits=3))",
        legend = false,
        color = :steelblue,
        alpha = 0.6,
        size = (800, 600),
        grid = true
    )
    
    #Adicionar linha de referência (y = x)
    min_val = min(minimum(valores_reais), minimum(predicoes))
    max_val = max(maximum(valores_reais), maximum(predicoes))
    plot!(scatter_plot, [min_val, max_val], [min_val, max_val], 
          color = :red, linestyle = :dash, linewidth = 2, 
          label = "Linha perfeita (y = x)")
    
    #Salvar scatter plot
    savefig(scatter_plot, "predicoes_vs_reais.png")
    println("✓ Scatter plot salvo em: predicoes_vs_reais.png")
    
    #Gráfico de resíduos
    residuos = valores_reais .- predicoes
    residuos_plot = scatter(
        predicoes,
        residuos,
        xlabel = "Valores Preditos (R\$)",
        ylabel = "Resíduos (R\$)",
        title = "Gráfico de Resíduos",
        legend = false,
        color = :orange,
        alpha = 0.6,
        size = (800, 600),
        grid = true
    )
    
    #Linha horizontal em y = 0
    plot!(residuos_plot, [minimum(predicoes), maximum(predicoes)], [0, 0], 
          color = :red, linestyle = :dash, linewidth = 2)
    
    savefig(residuos_plot, "residuos.png")
    println("✓ Gráfico de resíduos salvo em: residuos.png")
    
    #Histograma de resíduos
    histograma_residuos = histogram(
        residuos,
        bins = 50,
        xlabel = "Resíduos (R\$)",
        ylabel = "Frequência",
        title = "Distribuição dos Resíduos",
        legend = false,
        color = :purple,
        alpha = 0.7,
        size = (800, 600),
        grid = true
    )
    
    savefig(histograma_residuos, "histograma_residuos.png")
    println("✓ Histograma de resíduos salvo em: histograma_residuos.png")
    
catch e
    println("⚠ Erro ao gerar visualizações: $e")
end

#Comentar qualidade do modelo
println("\n[7/7] Análise da qualidade do modelo...")
println("\n" * "=" ^ 60)
println("Análise da qualidade do modelo:")
println("=" ^ 60)

if r2 >= 0.9
    println("✅ EXCELENTE: R² de $(round(r2, digits=3)) indica um ajuste muito bom do modelo.")
    println("   O modelo explica mais de 90% da variância nos preços de venda.")
elseif r2 >= 0.8
    println("✅ BOM: R² de $(round(r2, digits=3)) indica um bom ajuste do modelo.")
    println("   O modelo explica aproximadamente $(round(r2*100, digits=1))% da variância nos preços de venda.")
    println("   ⚠ Nota: Outliers podem afetar a performance, mas o modelo é útil para predições.")
elseif r2 >= 0.6
    println("⚠ MODERADO: R² de $(round(r2, digits=3)) indica um ajuste moderado do modelo.")
    println("   O modelo explica aproximadamente $(round(r2*100, digits=1))% da variância nos preços de venda.")
    println("   Sugestões de melhoria:")
    println("   - Feature engineering: criar novas features (ex: área total, área por banheiro)")
    println("   - Transformações: considerar log-transform para preços ou áreas")
    println("   - Outliers: investigar e possivelmente remover outliers extremos")
    println("   - Features adicionais: incluir mais variáveis explicativas se disponíveis")
elseif r2 >= 0.4
    println("⚠ BAIXO: R² de $(round(r2, digits=3)) indica um ajuste fraco do modelo.")
    println("   O modelo explica apenas $(round(r2*100, digits=1))% da variância nos preços de venda.")
    println("   Melhorias necessárias:")
    println("   - Feature engineering extensivo")
    println("   - Considerar modelos não-lineares (polinomiais, árvores de decisão)")
    println("   - Verificar multicolinearidade entre features")
    println("   - Análise de outliers e dados faltantes")
else
    println("❌ MUITO BAIXO: R² de $(round(r2, digits=3)) indica um ajuste muito fraco.")
    println("   O modelo precisa de revisão significativa.")
end

#Análise de outliers nas predições
residuos = valores_reais .- predicoes
residuos_abs = abs.(residuos)
outliers_idx = residuos_abs .> 3 * std(residuos)
n_outliers = sum(outliers_idx)

println("\n📊 Análise de outliers nas predições:")
println("   Outliers identificados (resíduos > 3σ): $n_outliers ($(round(100*n_outliers/length(predicoes), digits=1))%)")
if n_outliers > 0
    println("   RMSE sem outliers: $(round(sqrt(mean(residuos[.!outliers_idx].^2)), digits=2))")
    println("   ⚠ Outliers podem estar afetando a qualidade do modelo.")
end

#Integração com análises anteriores
println("\n" * "=" ^ 60)
println("Integração com análises anteriores:")
println("=" ^ 60)

#Comparar com histograma de preços
try
    if isfile("histograma_preco.png")
        println("✓ Histograma de preços disponível (histograma_preco.png)")
        println("  Compare a distribuição dos valores reais com as predições.")
    end
catch
end

#Comparar com correlações
try
    if isfile("correlacoes_resultados.json")
        println("✓ Resultados de correlação disponíveis (correlacoes_resultados.json)")
        println("  As features com maior correlação devem ter maior impacto no modelo.")
    end
catch
end

#Estatísticas comparativas
println("\n📊 Comparação estatística:")
println("   Diferença média entre real e predito: $(round(mean(residuos), digits=2)) R\$")
println("   Erro percentual médio: $(round(100*mean(abs.(residuos ./ valores_reais)), digits=2))%")
println("   Coeficiente de variação do RMSE: $(round(100*rmse/mean(valores_reais), digits=2))%")

#Sugestões de melhoria
println("\n" * "=" ^ 60)
println("Sugestões de melhoria:")
println("=" ^ 60)

if r2 < 0.8
    println("1. Feature Engineering:")
    println("   - Criar 'area_total' = area_primeiro_andar + area_segundo_andar")
    println("   - Criar 'area_por_banheiro' = area_total / quantidade_banheiros")
    println("   - Criar 'densidade_garagem' = capacidade_carros_garagem / area_total")
    
    println("\n2. Transformações:")
    println("   - Considerar log(preco_de_venda) como variável alvo")
    println("   - Aplicar normalização/standardização nas features")
    
    println("\n3. Modelos alternativos:")
    println("   - Regressão polinomial (features ao quadrado)")
    println("   - Random Forest ou Gradient Boosting")
    println("   - Regularização (Ridge, Lasso)")
    
    println("\n4. Análise de dados:")
    println("   - Investigar outliers específicos")
    println("   - Verificar interações entre features")
    println("   - Análise de multicolinearidade")
end

#Salvar métricas em arquivo
println("\n[8/8] Salvando resultados da avaliação...")

arquivo_avaliacao = "avaliacao_modelo.txt"
open(arquivo_avaliacao, "w") do f
    write(f, "Avaliação do Modelo de Regressão Linear\n")
    write(f, "=" ^ 50 * "\n\n")
    write(f, "Métricas:\n")
    write(f, "  MSE: $(round(mse, digits=2))\n")
    write(f, "  RMSE: $(round(rmse, digits=2))\n")
    write(f, "  MAE: $(round(mae, digits=2))\n")
    write(f, "  R²: $(round(r2, digits=4))\n\n")
    write(f, "Estatísticas:\n")
    write(f, "  Média valores reais: $(round(mean(valores_reais), digits=2))\n")
    write(f, "  Média predições: $(round(mean(predicoes), digits=2))\n")
    write(f, "  Outliers nas predições: $n_outliers\n")
    write(f, "  Erro percentual médio: $(round(100*mean(abs.(residuos ./ valores_reais)), digits=2))%\n")
end
println("✓ Resultados salvos em: $arquivo_avaliacao")

#Resumo final
println("\n" * "=" ^ 60)
println("Resumo da avaliação:")
println("=" ^ 60)
println("✓ Dataset de teste: $n_test observações")
println("✓ R²: $(round(r2, digits=4))")
println("✓ RMSE: $(round(rmse, digits=2)) R\$")
println("✓ Visualizações geradas:")
println("  - predicoes_vs_reais.png")
println("  - residuos.png")
println("  - histograma_residuos.png")
println("✓ Resultados salvos: $arquivo_avaliacao")

println("\n" * "=" ^ 60)
println("Avaliação do modelo concluída com sucesso!")
println("=" ^ 60)