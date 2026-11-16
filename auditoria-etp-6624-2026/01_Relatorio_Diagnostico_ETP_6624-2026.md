# RELATÓRIO DE AUDITORIA — FASES 1 A 5 (DIAGNÓSTICO E PLANO DE ALTERAÇÕES)

**Objeto auditado:** Estudo Técnico Preliminar — PA nº 6624/2026 — Secretaria Municipal de Educação de Bertioga (suporte, manutenção, plataformas e Programa Aluno Tutor de Tecnologia)
**Data da auditoria:** 16/11/2026
**Situação dos arquivos originais:** **NÃO ALTERADOS.** Este relatório contém apenas diagnóstico e propostas, que aguardam autorização expressa (Fase 5).

> Natureza do documento: análise técnico-administrativa simulada por especialidades. Não constitui parecer jurídico oficial, nem manifestação de procurador investido, controle interno ou Tribunal de Contas.

---

## CLASSIFICAÇÃO PRELIMINAR

### 🔴 NÃO CONFORME (versão "ETP_Finalizado_v1")

Motivos determinantes (detalhados nos itens C-01 e C-02):

1. **A Seção 6 (estimativa do valor) está sem nenhum valor**, embora o art. 18, § 2º, da Lei nº 14.133/2021 a torne elemento mínimo obrigatório (inciso VI). Mesmo assim, a Seção 13 declara "VIABILIDADE TÉCNICA E ECONÔMICA", uma conclusão sem suporte nos autos.
2. **As unidades de medida e as quantidades divergem** entre as Seções 4 e 6 e o Anexo IV (itens 3, 4 e 5). O próprio Anexo III aponta que essa divergência produz totais com diferença de até 40 vezes. O inciso IV (quantidades e memória de cálculo) também é elemento mínimo.

Há ainda 10 pontos de prioridade **Alta**, sobretudo competitividade, parcelamento, sobreposição com garantias e vantajosidade da vigência plurianual. Nenhum deles é intransponível, e o conteúdo técnico do ETP é, em grande parte, sólido e bem documentado.

---

## FASE 1 — INVENTÁRIO E LIMITAÇÕES

| # | Documento | Formato | Legibilidade | Observação |
|---|---|---|---|---|
| D1 | ETP_Finalizado_v1.md | Markdown, 366 linhas | Integral | Inclui os Anexos I a V. Sem paginação: a localização é feita por **seção e linha do arquivo** (Lnn). |
| D2 | Modelo_de_ETP_Prefeitura_de_Bertioga.md | Markdown, 263 linhas | Integral | Modelo de referência estrutural. |
| D3 | Lei_14133.md | Markdown, 477 linhas | Integral | **Atenção:** apenas os arts. 1º a 6º estão com texto integral. Os arts. 7º a 194 são **resumos/paráfrases** e não servem como fonte literal. |
| — | Anexo III (pesquisa de preços) | Contido no D1 | Integral | É só um **resumo executivo**. Não foram recebidos o relatório completo, as telas de busca, as cotações informais nem a planilha de cálculo. |
| — | Decreto Municipal nº 4.345/2023 | **Não recebido** | — | Localizado no sítio oficial de Bertioga, mas **inacessível** a partir deste ambiente (host bloqueado). |

**Limitações técnicas do ambiente (declaradas conforme o item 3 do prompt):**
- Acesso **bloqueado** (erro de proxy/DNS) a `bertioga.sp.gov.br` e `pncp.gov.br` (inclusive à API de busca). **Por isso, nenhuma consulta direta ao PNCP foi realizada.** Não declaro inexistência de contratações comparáveis no PNCP; declaro apenas que **não foi possível consultá-lo** daqui.
- A pesquisa foi feita por buscador web genérico, com acesso parcial a fontes secundárias. O portal de jurisprudência do TCU não foi aberto na íntegra.
- Não houve sistema multiagente real. As especialidades foram executadas em sequência, com registros separados e validação cruzada (Agente 7).
- Nenhuma ferramenta de terceiros foi instalada. As skills internas usadas foram `etp-tr-nllc` e `pesquisa-precos-licitacoes`.

---

## AGENTE 1 — MATRIZ DE CONFORMIDADE ESTRUTURAL (Modelo × ETP)

Legenda da natureza: **L** = exigência legal (art. 18, § 1º); **LM** = elemento mínimo obrigatório (art. 18, § 2º: incisos I, IV, VI, VIII e XIII); **R** = exigência do modelo municipal; **F** = orientação formal.

| Seção do modelo | Localização no ETP | Natureza | Situação | Evidência | Recomendação |
|---|---|---|---|---|---|
| Informações básicas (PA, unidade) | Seção 1, L14–15 | R | ✅ Atende | PA 6624/2026; SME | — |
| Introdução | Ausente | F | ⚠️ Ausente | — | Baixa: inserir o texto-padrão do modelo (B-22) |
| 1 – Descrição da necessidade (I) | Seção 1, L18–69 | **LM** | ✅ Atende, com ressalvas | Diagnóstico quantitativo (Anexo II) | Corrigir dados do Ideb (M-12) |
| 2 – Previsão no PCA (II) | Seção 2, L73–75 | L | ⚠️ Incompleta | Afirma alinhamento sem identificar o item do PCA | M-13 |
| 3 – Requisitos (III) | Seção 3, L79–99 | L | ⚠️ Atende, com riscos | Requisitos detalhados; alguns potencialmente restritivos | A-06, A-10 |
| 4 – Quantidades e memória de cálculo (IV) | Seção 4, L103–115 | **LM** | ❌ Desconforme | Tabela sem memória de cálculo; unidades incoerentes | C-02, A-04 |
| 5 – Levantamento de mercado (V) | Seção 5.1, L121–123 | L | ⚠️ Insuficiente | Só 2 alternativas, sem preço estimado por solução (o modelo exige) | A-07 |
| 6 – Estimativa do valor (VI) | Seção 6, L132–145 | **LM** | ❌ Desconforme | Todos os valores "A confirmar" / "R$ XXXXXX" | C-01 |
| 7 – Descrição da solução (VII) | Seção 7, L149–161 | L | ✅ Atende, com ressalvas | Três frentes e SLA | M-15 |
| 8 – Parcelamento (VIII) | **Seção 5.2**, L125–128 (local diferente do modelo) | **LM** | ⚠️ Insuficiente | Justificativa não cobre os itens 4 e 5 | A-03. O local diferente é aceitável. |
| 9 – Resultados pretendidos (IX) | Seção 8, L165–171 | L | ⚠️ Parcial | Sem indicadores mensuráveis (o modelo pede subsídio ao SLA/IMR) | M-15 |
| 10 – Providências prévias (X) | Seção 9, L175–182 | L | ⚠️ Atende, com inconsistência | Capacitação de fiscais não prevista; "habilitação" fora de fase | A-10 |
| 11 – Correlatas/interdependentes (XI) | Seção 11, L200–205 | L | ⚠️ Parcial | Não examina as garantias das aquisições de 2022–2025 | A-04 |
| 12 – Impactos ambientais (XII) | Seção 12, L209–213 | L | ✅ Atende | Logística reversa, PNRS | — |
| 13 – Gerenciamento de riscos | Seção 10, L186–196 | R | ⚠️ Parcial | 5 riscos. A nota remete o "Risco 6" à Seção 13, que não o trata | M-14 |
| Viabilidade (XIII) | Seção 13, L217–221 | **LM** | ❌ Desconforme quanto à parte econômica | Declara viabilidade econômica sem estimativa | C-01 |
| Anexo I modelo – Riscos (formato ficha) | Tabela na Seção 10 | F | ✅ Equivalente | A tabela cumpre a finalidade da ficha | Manter |
| Assinatura da equipe | Anexo I, L225–233 | R | 📝 Campo manual | — | Lista de providências manuais |

**Conclusão do Agente 1:** a ordem das seções difere do modelo (por exemplo, o parcelamento está em 5.2 e os riscos na Seção 10), mas **cumpre a finalidade**, e por isso não recomendo reordenar por razão estética. As falhas estruturais relevantes são materiais (Seções 4, 6 e 13), não de forma.

---

## AGENTE 2 — INCONSISTÊNCIAS TÉCNICAS E ADMINISTRATIVAS

| # | Inconsistência | Evidência (linha) | Consequência |
|---|---|---|---|
| T1 | Conclusão de viabilidade econômica sem estimativa | L138–143 × L221 | Motivação viciada (art. 18, § 1º, XIII). É o apontamento típico de controle externo. |
| T2 | Item 3: unidade "Mês", quantidade **1**, com vigência de 48 meses | L111, L140, L324 | Ambiguidade na fórmula de cálculo do total |
| T3 | Item 4: unidade "Mês (40 licenças)", quantidade **40**, na Seção 4; "Licença" no Anexo IV | L112, L141, L325 | É a causa do erro R$ 380.160 × R$ 9.504 apontado no próprio Anexo III (L293) |
| T4 | Item 5: unidade "Mês", quantidade **30**, embora o programa seja **anual** (30 alunos/ano); "Aluno/Turma" no Anexo IV | L113, L142, L326 | Total varia de R$ 57.000 a R$ 684.000 (L294) |
| T5 | Parque de 2.148 unidades coberto integralmente, mas o **lote 2025** (7 registros, "100% inoperantes") ainda deve estar em garantia; e "95,7% fora de garantia" refere-se só à amostra de 46 | L109, L266–267 | Risco de pagar pela manutenção de bens cobertos por garantia do fornecedor (duplicidade) |
| T6 | Seção 11 conclui "não há sobreposição", mas analisa só o PP 87/2021 (725 unidades). As aquisições de 2022–2023 e 2025 (≈1.423 unidades) não são examinadas | L202–205 × L42, L266 | A conclusão de não sobreposição é mais ampla do que a evidência |
| T7 | "Peças inclusas" sem exclusões (vandalismo, mau uso, sinistro, perda total) e sem limite | L81–82 | Risco de preço elevado por incerteza ou de proposta inexequível. Gera litígio na execução. |
| T8 | Seção 9, item 3, chama de "Habilitação Técnica" a entrega de certificações **e atestados** 2 dias úteis antes da assinatura | L181 × L90, L99 | O atestado de capacidade técnica é documento de **habilitação** (arts. 62 e 67) e deve ser apresentado na fase própria do certame. As certificações de equipe, ao contrário, estão corretamente postas como requisito de execução. |
| T9 | Seção 3(5)(iv): educador da rede acompanha. Anexo V: no trajeto, acompanham "servidor(es) da frota" | L89 × L351 | Responsabilidade de guarda dos alunos indefinida no deslocamento |
| T10 | "Seguro de acidentes **a terceiros**" para a visita de alunos | L88 | O seguro de responsabilidade perante terceiros **não cobre os próprios alunos**. O adequado é seguro de **acidentes pessoais coletivo** (ponto técnico de seguros a confirmar). |
| T11 | Ideb: "6,38 (resultado 2024)" na L26 × "6,3 em 2023" na L44; "segundo ano consecutivo" para um índice **bienal** | L26, L44 | Fragiliza a motivação. Ver M-12. |
| T12 | Plataforma (item 3) "consolida a administração via Google Admin Console", mas o próprio Anexo III manda confirmar a edição do Workspace "para evitar sobreposição", e o ponto não foi resolvido | L83, L295 | Risco de contratar funcionalidade já licenciada |
| T13 | Requisito de 9.920 acessos simultâneos sem memória de cálculo | L83 | Requisito sem lastro, que pode ser restritivo e afeta o preço |
| T14 | Severidades "Crítica" e "Alta" sem definição objetiva; sem vínculo com glosas (IMR) | L157–159 | O SLA não é exigível de forma objetiva |
| T15 | Anexo II: percentuais de defeito calculados sobre 46, mas o n declarado é 45 (15/45 = 33,3%, não 32,6%) | L249–251 | Erro aritmético menor |
| T16 | Anexo III cita "F1 a F17", mas nomeia só 11 órgãos, sem links nem datas | L289 | Rastreabilidade insuficiente da pesquisa |

**Conferências aritméticas confirmadas (corretas):** 725/2.148 = 33,75% ≈ 33,8% ✅ · 28+9+9 = 46 ✅ · 37/46 = 80,4% ✅ · 28/46 = 60,9% ✅ · 39/725 = 5,38% ≈ 5,4% ✅ · 30/39 = 76,9% ✅ · 20h + 70h = 90h; 90 × 30 = 2.700 horas-aluno ✅ · 30 × 4 = 120 alunos ✅ · R$ 198 × 40 × 48 = R$ 380.160 ✅ · R$ 475 × 30 × 48 = R$ 684.000 ✅ · R$ 475 × 30 × 4 = R$ 57.000 ✅.

---

## AGENTE 3 — CONFORMIDADE NORMATIVA

Classificação: ✅ Confirmada por evidência · 🟡 Sustentada com ressalvas · 🔵 Depende de informação complementar · ⚫ Não verificável com os recursos disponíveis.

| Dispositivo citado | Local no ETP | Fonte consultada | Conclusão | Correção recomendada |
|---|---|---|---|---|
| Lei 14.133/2021, art. 18, § 1º, I a XIII e § 2º | Cabeçalho; Seções 1 e 12 | D2 (transcrição no modelo) e [Planalto](https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm) | ✅ Pertinente. O § 2º torna obrigatórios os incisos I, IV, VI, VIII e XIII. | Nenhuma no texto; corrigir as Seções 4 e 6 |
| Art. 23, § 1º, IV (mín. 3 fornecedores; orçamentos com até 6 meses) | Seção 6, L134 | D2 (transcrição) | ✅ Correto. Atenção: o próprio inciso exige **justificativa da escolha dos fornecedores**, que o ETP não apresenta. | Incluir critério de seleção dos fornecedores (A-07) |
| Art. 106, II (descontinuidade orçamentária) | Seção 1, L66; Seção 10, L196 | Fontes secundárias ([Justen](https://justen.com.br/wp-content/uploads/2023/06/IE182PrazoContratualLei14133.pdf), [Zênite](https://zenite.blog.br/nova-lei-de-licitacoes-prazo-inicial-e-prorrogacao-dos-contratos-de-servicos-continuos/)); Planalto não aberto na íntegra | 🟡 O inciso II trata do **atestado anual** de créditos. A extinção sem ônus por falta de créditos é o **inciso III** (redação a conferir no Planalto). **Omissão relevante:** o **inciso I** exige atestar a "maior vantagem econômica" da contratação plurianual, e isso não foi demonstrado para 48 meses. | A-08 / M-17 |
| Art. 106 (vigência de 48 meses ≤ 5 anos) | Seção 13, L220 | Idem | ✅ Dentro do limite | — |
| Art. 6º, XIII e XLI (serviço comum; pregão) | Seção 13, L219 | D3 (arts. 1º a 6º com texto integral) | ✅ Dispositivos corretos. 🟡 Falta demonstrar que **todos** os itens são "comuns" (o item 5 é formação pedagógica, cf. art. 6º, XVIII, "f") | M-21 |
| Art. 92 (reajuste no contrato) | Seção 13, L220 | D3 (resumo) | 🟡 Pertinente, mas o art. 25, § 7º, exige que o **edital** fixe o índice. "Índice setorial (ex.: IPCA)" é impreciso, porque o IPCA não é índice setorial. | M-16 |
| Art. 18, § 1º, XII; Lei 12.305/2010 | Seção 12 | D2 | ✅ Pertinente | — |
| Súmula TCU 247 | Seção 5.2 | Fontes secundárias ([TCU – Licitações e Contratos 4.1.8](https://licitacoesecontratos.tcu.gov.br/4-1-8-justificativas-para-o-parcelamento-ou-nao-da-contratacao), [TCDF](https://jurisprudencia.tc.df.gov.br/wp-content/uploads/2018/07/Objeto-divisível.-Adjudicação-por-item.-Súmula-TCU-247.pdf)) | 🟡 A súmula existe e é pertinente, mas **a regra é a adjudicação por item**. O ETP cita a súmula para sustentar o oposto (lote único), sem demonstrar a inviabilidade para os itens 4 e 5. Não vincula o Município; tem valor de referência. | A-03 |
| Súmula TCU 272 | Seção 3(7), L99 | Fontes secundárias ([TCDF](https://jurisprudencia.tc.df.gov.br/wp-content/uploads/2018/07/Custos-antecipados-para-as-licitantes.-Súmula-TCU-272.pdf), [Zênite](https://zenite.blog.br/topico/credenciamento-pelo-fabricante/)) | ✅ Pertinente e bem aplicada. Também fundamenta deslocar as certificações para a fase pré-contratual. | Citá-la também no requisito 6 |
| Acórdão TCU nº 3.569/2023 – 2ª Câmara | Seção 6, L145 | [Blog Zênite](https://zenite.blog.br/vicios-no-planejamento/) (secundária) | ⚫/🟡 O acórdão existe (Rel. Min. Marcos Bemquerer, 23/05/2023, segundo a fonte secundária). O trecho localizado trata da **possível inviabilidade de pesquisa em sistemas públicos**, e **não** da tese de que "valores preliminares informais não devem fundamentar a publicação do edital". A atribuição **não foi confirmada**. | A-11: retirar a atribuição ou conferir o inteiro teor no portal do TCU |
| Decreto Municipal nº 4.345/2023, art. 14, § 3º e art. 8º | Seção 6, L134; Anexo I, L227 | [Sítio oficial de Bertioga](https://www.bertioga.sp.gov.br/wp/wp-content/uploads/2024/01/DECRETO-4345-27-DE-DEZEMBRO-DE-2023-REGULAMENTACAO-LEI-14133-DE-2021.pdf), localizado mas **inacessível** | 🔵 Confirmada a existência e o objeto (regulamenta a Lei 14.133/2021, datado de 27/12/2023). **Não confirmados** a redação do art. 14, § 3º, a do art. 8º e a vigência atual. | Conferência manual obrigatória (lista de providências) |
| Lei 14.533/2023 (Política Nacional de Educação Digital) | Seção 1, L52 | Conhecimento consolidado; não reconsultada | 🟡 Número e tema corretos | — |
| LGPD (Lei 13.709/2018) | Seções 1, 7 e 10 | — | ✅ Pertinente. 🟡 Falta referência ao **art. 14** (dados de crianças e adolescentes) no Programa Aluno Tutor e no Anexo V. | M-19 |
| Lei Complementar nº 220/2025 | Seção 8, L171 | [Câmara – Legislação Informatizada](https://www2.camara.leg.br/legin/fed/leicom/2025/leicomplementar-220-31-outubro-2025-798216-norma-pl.html) | ✅ Existe (31/10/2025; institui o SNE). Cria a Infraestrutura Nacional de Dados da Educação (Inde) e trata de interoperabilidade e LGPD. Pertinente. | — |
| Portaria MEC nº 269/2026 (EducaDados) | Seção 8, L171 | [Agência Gov](https://agenciagov.ebc.com.br/noticias/202603/startup-gov-br-gestao-e-mec-lancam-plataforma-nacional-de-dados-da-educacao), [Congresso em Foco](https://www.congressoemfoco.com.br/artigo/117658/governanca-de-dados-para-qualidade-da-gestao-e-equidade-educacional) | ⚫ O EducaDados/Inde existe e foi regulamentado em 24/03/2026, mas **o número 269/2026 não foi confirmado** | A-11: conferir no DOU |
| PP nº 87/2021 (PA 8210/2021) | Seção 11, L202 | Não localizado | ⚫ Dado interno, não verificável | Conferência manual |
| Ideb: Bertioga 6,38 / 6,6 / 1º lugar | Seção 1, L26, L44 | [Diário do Litoral](https://www.diariodolitoral.com.br/bertioga/bertioga-tornou-se-a-cidade-da-oportunidade-e-do/208655/); [JB](https://www.jb.com.br/brasil/educacao/2026/08/1060517-ideb-2025-registra-maior-evolucao-acumulada-em-20-anos.html) | ⚫ Confirmada apenas a **declaração do prefeito** sobre o 1º lugar na Baixada "pelo segundo ano consecutivo". A nota 6,6 de Bertioga **não foi localizada** (6,6 é a nota do Estado de SP nos anos iniciais em 2025). | M-12: anexar a consulta ao Inep |

**Registro de integridade:** nenhuma redação legal foi reconstruída neste relatório. As remissões a dispositivos foram feitas por número, com a interpretação separada. Os trechos entre aspas vêm do D2 (transcrição do art. 18 e do art. 23 no modelo municipal) ou de fontes indicadas.

---

## AGENTE 4 — PESQUISA DE PREÇOS E COMPARABILIDADE

### Etapa A — Decomposição econômica do objeto (confirmada documentalmente)

| Item | Direcionador de custo | Componentes (com base no ETP) | Unidade de faturamento coerente (proposta) |
|---|---|---|---|
| 1 | Parque (2.148) × taxa de falha × custo de peça + equipe fixa | NOC 8×5 (web, e-mail, telefone), N1 remoto, N2 presencial, preventiva bimestral, peças **sem limite** (L81) | **dispositivo/mês** × 2.148 × 48 |
| 2 | 59 estações | Preventiva bimestral, peças (rodas, fechaduras, contatos), tratamento anticorrosivo (L82) | **estação/mês** × 59 × 48 |
| 3 | Assinatura de plataforma + serviço | Nuvem, 9.920 acessos, IA, telemetria ≤ 10 min, privacidade diferencial (L83) | **serviço/mês** × 1 × 48 (qtd. = 48 meses) |
| 4 | Licenças | MDM, 40 tablets, geolocalização, restrição de apps (L84) | **licença/mês** × 40 × 48 |
| 5 | Turmas anuais | 90h por turma, materiais, alimentação, formatura, seguro (L85–89) | **aluno/ano** × 30 × 4 = 120 (ou **turma/ano** × 4) |

### Etapas B, C e D — Fontes e comparabilidade

**Pesquisas efetivamente realizadas nesta auditoria (16/11/2026):**

| # | Plataforma | Termos | Resultado |
|---|---|---|---|
| P1 | API de busca do PNCP (`pncp.gov.br/api/search`) | "chromebook"; "manutencao chromebook"; "aluno tutor" | **Falha de acesso (proxy 403).** Nenhum dado obtido. |
| P2 | Buscador web | "PNCP manutenção Chromebook suporte técnico prefeitura contrato 2025" | Nenhum objeto específico de manutenção de Chromebook. Resultados genéricos de TI (Niterói, Timon etc.), 🔴 não comparáveis. |
| P3 | Buscador web | "TRT 16ª Região pregão eletrônico 90011/2025 MDM…" | **Não localizado.** O nº 90011/2025 aparece em outros órgãos (TRE-MT, TJPB, TRT-5) com outros objetos. |

**Validação das referências do Anexo III:**

| Ref. | Fonte | Objeto / Unidade | Valor | Comparabilidade | Decisão |
|---|---|---|---|---|---|
| TRT-16, PE 90011/2025 | Indicada no Anexo III; **não localizada** por esta auditoria | MDM, 4.140 licenças, licença/mês | média de R$ 4,93 | 🟡 Parcial: escala 100× maior, sem SLA 8×5, sem geolocalização, sem SecurityScorecard | **Manter com ressalva**, condicionada à juntada do link do PNCP, da data, da natureza do valor (estimado, adjudicado ou contratado) e da tela de consulta |
| F1–F17 (TRT-16, UERN, SEDUC-GO, Contenda/PR, Santo Antônio da Patrulha/RS, Borda da Mata/MG, Piracicaba, Indaiatuba, Americana, Carapicuíba, Guaratinguetá) | Anexo III | Hardware, licenciamento avulso, programas pedagógicos | — | 🔴 Inadequadas segundo o próprio anexo, por objeto distinto | **Manter só como registro de pesquisa negativa**, com link, data e motivo da exclusão. **Faltam 6 das 17 fontes.** |
| Valor preliminar item 4: R$ 198,00 | Informal (origem não documentada) | ? | R$ 198,00 | — | **Achado analítico:** R$ 198 ÷ 40 = **R$ 4,95/licença/mês**, praticamente igual à média do TRT-16 (R$ 4,93). **Hipótese forte (não fato):** R$ 198 é o **total mensal** das 40 licenças. Nesse caso, o total correto em 48 meses é **R$ 9.504,00**, e R$ 380.160,00 é multiplicação duplicada. Confirmar com o fornecedor. |
| Valor preliminar item 5: R$ 475,00 | Informal | ? | R$ 475,00 | — | Indeterminado. Se for por aluno/mês, dá R$ 171.000/ano (R$ 5.700/aluno/ano). Se for por aluno/ano, dá R$ 14.250/ano. **Exige esclarecimento formal por escrito.** |

### Etapa E — Ausência de referências diretamente comparáveis

O que a auditoria pode afirmar: **não foi possível localizar**, pelos meios disponíveis, contratação pública diretamente comparável para os itens 1, 2, 3 e 5. Isso **não equivale** a afirmar que tais contratações não existem.

**Roteiro para tornar a justificativa defensável:**
1. **PNCP** (feito pela equipe, no navegador): buscar em "Editais" e "Contratos", com filtro de 12 meses (art. 23, § 1º, II), os termos: `chromebook`; `manutenção de chromebook`; `assistência técnica chromebook`; `carrinho de recarga` / `estação de carregamento` / `armário de carregamento`; `manutenção notebook educacional`; `MDM`; `gerenciamento de dispositivos móveis`; `Google Workspace for Education suporte`; `aluno tutor` / `formação de alunos tecnologia`. Salvar **todas as telas com data e hora**, inclusive as de resultado zero, e abrir os **TRs** e as **planilhas de itens**, não só os títulos.
2. **Painel de Preços / Compras.gov.br:** os mesmos termos, com os códigos CATSER pertinentes (manutenção de microcomputador portátil; licença de software como serviço).
3. **TCE-SP (Audesp/painéis de transparência):** contratações de municípios paulistas.
4. **Cotação direta (art. 23, § 1º, IV):** **no mínimo 3** fornecedores, com **justificativa escrita da escolha** de cada um (exigência expressa do inciso), usando **o mesmo formulário com unidades corrigidas** (C-02) e **planilha de composição de custos obrigatória**. Recomendação: incluir fornecedores de **perfis distintos**, como integradora parceira Google, assistência técnica multimarca e fornecedor de MDM puro, para reduzir o risco de cesta "viciada" em um único modelo de negócio.
5. **Estimativa por componentes (item 1),** como verificação de razoabilidade das cotações, e **não** como substituta delas:

```
Custo anual de peças (CP) = Σ_k [ N_parque × TFA_k × Preço_peça_k ]
   k = tela, teclado, bateria, placa/sistema, outros
   TFA_k = taxa de falha anual por componente (derivar do histórico interno: hoje só há
           46 registros; a distribuição observada é tela 33%, teclado 31%, sistema 18%,
           bateria 13%, outros 4%; a taxa global anualizada AINDA NÃO FOI CALCULADA)
   Preço_peça_k = cotação de peças (fonte a documentar)
Custo anual de mão de obra (CM) = (horas N1 + horas N2 + visitas preventivas × deslocamento) × custo-hora
Preço dispositivo/mês ≈ (CP + CM) × (1 + BDI/margem) ÷ 12 ÷ 2.148
```
[VALIDAÇÃO NECESSÁRIA] Nenhum preço de peça ou custo-hora foi pesquisado ou inserido aqui.

### Etapa F — Metodologia × art. 23 e regulamento municipal

- A cotação direta é parâmetro **admitido** pelo art. 23, § 1º, IV. A lei usa "adotados de forma combinada ou não", e por isso não há hierarquia obrigatória entre os incisos no texto legal (eventual hierarquia pode constar do decreto municipal, **não verificado**).
- A "validade máxima de 90 dias" (L298, L315) é mais restritiva que o limite legal de 6 meses. É legítima se vier do Guia municipal, mas **o Guia não foi recebido**.
- **Não se presume** aplicação da IN SEGES/ME 65/2021 ao Município. Ela pode ser citada como **boa prática** (média, mediana, menor preço; descarte de inexequíveis e excessivos).
- Método estatístico sugerido: **mediana** se houver dispersão (CV > 25%) e **média** se a amostra for homogênea, **sempre com justificativa**. Com só 3 cotações, recomenda-se registrar o CV e, se for alto, buscar uma 4ª e uma 5ª cotação.

**Classificação da pesquisa de preços (situação atual):** 🔴 **Insuficiente.** Não há cotações formais, as unidades são incoerentes e as fontes não estão documentadas. Isso é coerente com a própria advertência do ETP. A metodologia **proposta** é defensável.

---

## AGENTE 5 — PARECER TÉCNICO-JURÍDICO SIMULADO

**Ementa:** ETP de serviços continuados de TI educacional. Lote único com 5 itens heterogêneos, vigência de 48 meses, pregão com menor preço. Faltam elementos mínimos (art. 18, § 2º). Há riscos de restrição à competitividade. **Prosseguimento condicionado.**

**I. Irregularidades comprovadas (documentais):**
1. Ausência de estimativa do valor (art. 18, § 1º, VI e § 2º). O próprio modelo municipal (D2, Seção 6) esclarece que a estimativa do ETP **é preliminar e se distingue da pesquisa de preços do art. 23**. Portanto, **não é preciso aguardar o mapa formal** para preencher a Seção 6. É preciso, sim, apresentar uma estimativa preliminar fundamentada e rotulada como tal.
2. Quantidades sem memória de cálculo e com unidades incoerentes (art. 18, § 1º, IV).
3. Conclusão de viabilidade econômica sem suporte (motivação, art. 5º).

**II. Riscos potenciais (dependem da forma final do TR/edital):**
1. **Restrição à competitividade:** são exigidos 18 profissionais com credenciais de um único fabricante, incluindo **4 "Google Innovator"** (programa de seleção do próprio Google para educadores, cuja oferta depende do fabricante) e **2 "Professional Cloud Architect"** (pertinência à manutenção de Chromebooks não demonstrada). Somam-se SecurityScorecard "A", disponibilidade "comprovada por órgão independente", privacidade diferencial e IA conversacional, e uma visita que depende da anuência de empresa privada (Google Brasil). O **conjunto**, mais do que cada requisito isolado, pode reduzir o universo a poucos parceiros do fabricante. A menção a precedentes em Indaiatuba e Itapevi, sem documentação, aumenta a percepção de modelo pré-definido. Mitigação: justificar a pertinência requisito a requisito, aceitar **equivalentes**, dimensionar a equipe pela demanda e manter tudo na fase de execução (Súmula 272).
2. **Lote único:** os itens 4 (MDM para 40 tablets) e 5 (formação de alunos) têm mercado próprio. A justificativa da Seção 5.2 só demonstra interdependência entre os itens 1, 2 e 3. Há risco de impugnação.
3. **Duplicidade com garantias** dos lotes mais recentes.
4. **Vigência plurianual sem o atestado do art. 106, I.**
5. **Pregão para o item 5:** é admissível se o serviço for padronizável (art. 6º, XIII), mas convém justificar expressamente, porque a formação se aproxima do art. 6º, XVIII, "f".
6. **Dados de crianças (LGPD, art. 14)** no Programa Aluno Tutor e na plataforma de governança: falta base legal e falta cláusula de tratamento e portabilidade ao término do contrato.
7. **Segregação de funções (art. 7º, § 1º):** o servidor que elabora o ETP é também indicado como gestor do contrato. Não há vedação expressa nesse caso, mas recomendo submeter o ponto à assessoria jurídica e ao controle interno.

**III. Pontos que dependem de manifestação administrativa:** a redação vigente do Decreto 4.345/2023 (arts. 8º e 14); o item do PCA; a edição do Google Workspace contratada; as datas de garantia por lote; a origem e a natureza dos valores R$ 198 e R$ 475; os dados oficiais do Ideb.

**Conclusão simulada:** o ETP **não está apto** a embasar o TR na versão v1. Fica apto após o saneamento dos itens Críticos e Altos do plano abaixo, sem prejuízo da análise da Procuradoria (art. 53).

---

## AGENTE 7 — CONTRADITÓRIO E VALIDAÇÃO INDEPENDENTE

| Conclusão revisada | Contestação | Resultado |
|---|---|---|
| "Seção 6 vazia = Crítica" | Poderia ser só "pendência", já que o próprio ETP adverte sobre o mapa de preços | **Mantida como Crítica.** O § 2º do art. 18 torna o inciso VI obrigatório, e a Seção 13 já declara viabilidade econômica. A pendência só seria aceitável se a conclusão também fosse condicional. |
| "R$ 198 é o total mensal" (Agente 4) | É inferência pela coincidência com o TRT-16 | **Rebaixada a hipótese.** Mantida apenas como pergunta ao fornecedor. |
| "Requisitos restritivos" (Agente 5) | O Município é referência Google e o ecossistema é Google, o que justifica conhecimento específico | **Parcialmente acolhida.** É legítimo exigir domínio do Google Admin/Workspace. O que é questionável são a quantidade (18), a credencial "Innovator" (que mede inovação pedagógica docente, não suporte técnico) e a ausência de "ou equivalente". |
| "Lote único irregular" | O TCU admite o agrupamento quando não há mercado próprio por item | **Divergência pendente.** Se a equipe demonstrar com dados de mercado (cotações) que fornecedores de MDM ou de formação não atuam isoladamente, o lote se sustenta. Por isso o plano oferece **duas opções** (A-03). |
| "Acórdão 3.569/2023 mal atribuído" | A fonte é secundária e parcial | **Rebaixada a "não confirmada".** A recomendação é conferir, e não afirmar erro. |
| "Ideb incorreto" | Não houve acesso ao Inep | **Mantida só como inconsistência interna** (6,38 × 6,3; "segundo ano" × índice bienal). O valor correto depende da consulta oficial. |
| Conflito entre recomendações | A-06 (reduzir requisitos) × Risco 3 da matriz (falta de equipe qualificada) | **Resolvido:** reduzir as exigências **de habilitação**, mas manter as exigências **de execução** com prazo para comprovar e substituição por equivalente. |

---

## AGENTE 6 — PLANO CONSOLIDADO DE ALTERAÇÕES (FASE 5 — AGUARDA APROVAÇÃO)

> Localização: D1 = ETP_Finalizado_v1.md, por seção e linha (Lnn). Nenhuma alteração será aplicada sem autorização.

### Tabela-resumo

| ID | Documento e localização | Problema | Fundamentação | Correção sugerida | Prioridade |
|---|---|---|---|---|---|
| **C-01** | D1, Seções 6 (L132–145) e 13 (L221) | Estimativa vazia; viabilidade econômica declarada sem base | Art. 18, § 1º, VI, XIII e § 2º; modelo (D2, Seção 6) | Inserir estimativa **preliminar** rotulada e tornar a conclusão coerente | **Crítica** |
| **C-02** | D1, Seção 4 (L107–113), Seção 6 (L136–143), Anexo IV (L320–326) | Unidades e quantidades incoerentes; sem memória de cálculo | Art. 18, § 1º, IV e § 2º | Padronizar unidades e acrescentar memória de cálculo | **Crítica** |
| A-03 | D1, Seção 5.2 (L125–128) | Lote único não justificado para os itens 4 e 5 | Art. 18, § 1º, VIII; art. 47, II (parcelamento); Súmula TCU 247 | Opção A (parcelar) **ou** Opção B (reforçar a justificativa) | Alta |
| A-04 | D1, Seção 4 e Seção 11 (L200–205) | Possível sobreposição com garantias vigentes | Art. 18, § 1º, IV e XI; economicidade | Segregar a frota por situação de garantia | Alta |
| A-05 | D1, Seção 3, requisitos 1 e 2 (L81–82) | Peças "inclusas" sem exclusões nem regra de perda total | Art. 18, § 1º, III e VII | Definir exclusões e laudo de irreparabilidade | Alta |
| A-06 | D1, Seção 3, requisitos 3 a 6 (L83–98) | Exigências potencialmente restritivas | Art. 9º, I; art. 5º (competitividade); Súmula 272 | Justificar pertinência; aceitar "ou equivalente"; ajustar quantidades | Alta |
| A-07 | D1, Seção 5.1 (L121–123) | Levantamento de mercado com só 2 alternativas e sem preço | Art. 18, § 1º, V; modelo (D2, Seção 5) | Ampliar alternativas, com custo e comparação | Alta |
| A-08 | D1, Seção 13 (L220) | Vantajosidade de 48 meses não atestada | Art. 106, I | Inserir motivação da vigência plurianual | Alta |
| A-09 | D1, Seção 3, requisito 3 (L83) | Sobreposição com o Workspace não resolvida; 9.920 sem memória | Art. 18, § 1º, IV e XI | Inserir verificação da edição e memória do dimensionamento | Alta |
| A-10 | D1, Seção 9, item 3 (L181) | Atestados tratados como pré-contratuais | Arts. 62, 63 e 67; Súmula 272 | Separar habilitação (atestados) de execução (certificações) | Alta |
| A-11 | D1, L145, L171, L134 e L289–290 | Citações não confirmadas (Acórdão 3.569/2023; Portaria MEC 269/2026; art. 14, § 3º do Decreto; TRT-16) | Rastreabilidade; art. 5º (motivação) | Conferir e completar ou retirar | Alta |
| M-12 | D1, Seção 1 (L26, L44) | Dados do Ideb inconsistentes | Motivação | Uniformizar com a fonte do Inep | Média |
| M-13 | D1, Seção 2 (L73–75) | PCA genérico | Art. 12, VII; art. 18, § 1º, II | Identificar o item do PCA ou justificar a ausência | Média |
| M-14 | D1, Seção 10 (L186–196) | Matriz de riscos incompleta; nota remete a seção que não trata o risco | Modelo (D2, Seção 13) | Acrescentar os Riscos 6 a 11 | Média |
| M-15 | D1, Seções 7 (L157–159) e 8 (L165–171) | SLA sem severidades definidas; resultados sem indicadores | Art. 18, § 1º, VII e IX; modelo (D2, Seção 9) | Definir severidades e indicadores | Média |
| M-16 | D1, Seção 13 (L220) | "Índice setorial (ex.: IPCA)" impreciso | Art. 25, § 7º; art. 92 | Fixar o índice ou remeter ao TR com critério | Média |
| M-17 | D1, Seção 1 (L66) e nota da L196 | Remissão ao art. 106, II, quando a extinção está no inciso III | Art. 106 | Ajustar a remissão | Média |
| M-19 | D1, Seção 3(5) (L85–89) e Anexo V (L333–366) | Seguro inadequado; LGPD de menores; acompanhamento no trajeto | LGPD, art. 14; coerência interna | Ajustar o requisito e o termo | Média |
| M-20 | D1, Anexo III (L286–298) | F1–F17 incompletas, sem link e sem data | Art. 23; rastreabilidade | Completar a matriz de fontes | Média |
| M-21 | D1, Seção 13 (L219) | Natureza "comum" do item 5 não demonstrada | Art. 6º, XIII; art. 29 | Inserir justificativa | Média |
| B-18 | D1, Anexo II (L249–254) | Percentuais com denominador 46 em vez de 45 | Exatidão | Recalcular | Baixa |
| B-22 | D1, após L10 | Introdução do modelo ausente | Modelo (D2) | Inserir introdução padrão | Baixa |
| B-23 | D1, Seção 9 (L178) | Elaborador do ETP também é gestor | Art. 7º, § 1º (segregação) | Submeter à assessoria jurídica; sem alteração de texto | Baixa (informativa) |

### Propostas de nova redação (integrais)

---

**C-01 — Seção 6 (substitui o parágrafo da L134, a tabela L136–143 e a advertência L145)**

*Trecho original (L134):* "A estimativa observou o art. 23, § 1º, da Lei nº 14.133/2021 e o art. 14, § 3º, do Decreto Municipal nº 4.345/2023. Como a pesquisa em fontes públicas abertas [...] exigindo decomposição de custos (peças nos itens 1 e 2; custos por turma/aluno no item 5)."

*Nova redação:*
> **6.1. Natureza da estimativa.** A estimativa apresentada nesta seção tem caráter **preliminar**, destina-se à avaliação da viabilidade econômica da solução escolhida (art. 18, § 1º, VI, da Lei nº 14.133/2021) e não se confunde com o valor de referência da licitação. Este será fixado no Termo de Referência a partir da pesquisa de preços formal de que trata o art. 23 da mesma Lei, documentada no Anexo III.
>
> **6.2. Fontes e método.** A pesquisa em bancos públicos (PNCP e Painel de Preços), documentada no Anexo III, não localizou contratações tecnicamente comparáveis para os itens 1, 2, 3 e 5. Para o item 4, foi localizada uma referência parcialmente comparável. A estimativa preliminar baseia-se em [**preencher: nº de orçamentos preliminares, fornecedores e datas**], normalizados às unidades da Seção 4, e será substituída pela cotação formal com no mínimo 3 (três) fornecedores (art. 23, § 1º, IV), acompanhada da justificativa de escolha de cada fornecedor.
>
> **6.3. Quadro de estimativa preliminar.** *(tabela com as unidades de C-02; valores a preencher a partir das propostas preliminares, sem invenção de dados)*
>
> **6.4. Disponibilidade orçamentária.** A despesa estimada para o exercício de [**ano**] correrá à conta da dotação [**preencher**]. Para os exercícios seguintes, a continuidade depende de previsão no Plano Plurianual e nas leis orçamentárias, com atestado anual da existência de créditos (art. 106, II).

*Nova redação da Seção 13, "Declaração de Viabilidade" (L221):*
> Conclui-se pela **VIABILIDADE TÉCNICA** da contratação. A **viabilidade econômica** é demonstrada, em caráter preliminar, pela estimativa da Seção 6, e será confirmada no Termo de Referência após a conclusão da pesquisa de preços formal (Anexo III). O prosseguimento fica condicionado a essa confirmação.

*Advertência (L145):* retirar a atribuição ao "Acórdão TCU nº 3.569/2023 – 2ª Câmara", a menos que a equipe confirme no inteiro teor que o acórdão sustenta a tese; manter a advertência como orientação da própria equipe de planejamento.

---

**C-02 — Seção 4 (substitui a tabela L107–113 e a nota L115) e Anexo IV (L320–326)**

> | Item | Descrição | Unidade de medida | Quantidade | Período | Fórmula do total |
> |---|---|---|---:|---:|---|
> | 1 | Suporte técnico NOC 8×5 e manutenção de Chromebooks, com peças | dispositivo/mês | 2.148 | 48 meses | P1 × 2.148 × 48 |
> | 2 | Suporte e manutenção preventiva do sistema de carregamento, com peças | estação/mês | 59 | 48 meses | P2 × 59 × 48 |
> | 3 | Plataforma de governança de dados educacionais | serviço/mês | 1 | 48 meses | P3 × 1 × 48 |
> | 4 | Solução MDM para tablets | licença/mês | 40 | 48 meses | P4 × 40 × 48 |
> | 5 | Programa Aluno Tutor de Tecnologia | aluno/ano | 30 por ano (120 no total) | 4 ciclos anuais | P5 × 30 × 4 |
>
> **Memória de cálculo.**
> (a) **Item 1:** 2.148 Chromebooks do inventário patrimonial em [**data**]. Os equipamentos ainda cobertos por garantia contratual do fornecedor serão **incluídos na cobertura apenas a partir do término da respectiva garantia**, conforme o quadro de garantias por lote (Seção 11). [Ver A-04: se a equipe preferir, a quantidade inicial passa a ser só a frota fora de garantia, com inclusões progressivas.]
> (b) **Item 2:** 59 estações (20 do PP 87/2021 e 39 de [**origem a preencher**]).
> (c) **Item 3:** 1 ambiente. O limite de 9.920 acessos simultâneos corresponde a [**memória: nº de alunos + docentes + gestores com conta institucional**].
> (d) **Item 4:** 40 tablets incompatíveis com o Google Admin Console (inventário em [**data**]).
> (e) **Item 5:** 30 alunos por ciclo anual. Dimensionamento conservador em relação à média histórica de 40 alunos/ano (2023–2025), com 4 ciclos na vigência de 48 meses, totalizando 120 alunos. O preço P5 é **por aluno e por ciclo anual completo** (90 horas, materiais, alimentação, formatura e seguro).

*No Anexo IV,* ajustar a coluna "Unid." às unidades acima e substituir as colunas de preço por: "Preço unitário (na unidade indicada)", "Total anual" e "Total 48 meses", com esta instrução ao proponente: "**Para o item 5, o preço unitário refere-se a um aluno em um ciclo anual completo. Para o item 4, refere-se a uma licença por mês.**"

---

**A-03 — Seção 5.2 (L125–128): duas opções para decisão do gestor**

*Opção A (parcelamento, recomendada pelo menor risco de impugnação):*
> Em observância ao princípio do parcelamento (art. 47, II, e art. 18, § 1º, VIII, da Lei nº 14.133/2021) e à Súmula TCU nº 247, a licitação será organizada em: **Grupo 1 — itens 1, 2 e 3**, adjudicados em conjunto em razão da interdependência operacional (administração dos Chromebooks no Google Admin Console integrada à plataforma de governança, Service Desk único e logística comum das visitas bimestrais aos mesmos equipamentos e estações); **Item 4 — MDM**, adjudicação autônoma, por existir mercado próprio de fornecedores de MDM; **Item 5 — Programa Aluno Tutor**, adjudicação autônoma, por se tratar de serviço de formação com mercado próprio.

*Opção B (manter lote único, com reforço da motivação):* acrescentar à redação atual:
> III. Quanto aos itens 4 e 5, a consulta ao mercado (Anexo III) demonstrou [**preencher com evidência: ex. "que os fornecedores consultados de MDM e de formação não ofertam esses serviços de forma isolada para o porte desta contratação" ou "que a contratação isolada do item 4, de valor estimado de R$ [x], geraria custo administrativo de gestão contratual desproporcional"**]. Por isso, a adjudicação autônoma não ampliaria a competitividade e acarretaria perda de economia de escala.
>
> *Observação do auditor: a Opção B só se sustenta com essa evidência documentada. Sem ela, recomenda-se a Opção A.*

---

**A-04 — Seção 11, novo item 3 (após L203)**

> 3. **Garantias contratuais vigentes dos Chromebooks adquiridos entre 2022 e 2025:** [**preencher por lote: processo, quantidade, data de recebimento, prazo de garantia, término**]. Os equipamentos em garantia permanecem sob responsabilidade do respectivo fornecedor até o término do prazo, e a contratada só passará a atendê-los após esse término, mediante inclusão formal na base de cobertura, de modo a afastar pagamento em duplicidade. Os 7 (sete) equipamentos do lote 2025 registrados como inoperantes (Anexo II, item 6) devem ser encaminhados de imediato à garantia do fornecedor.

---

**A-05 — Seção 3, requisito 1 (acrescentar ao final da L81)**

> Não se incluem na cobertura sem custo: (i) danos decorrentes de vandalismo, uso indevido comprovado em laudo, sinistro (incêndio, inundação, furto) ou força maior; e (ii) equipamentos declarados **irrecuperáveis** em laudo técnico circunstanciado, aprovado pela fiscalização, os quais serão baixados da base de cobertura com o correspondente ajuste do quantitativo faturado. Nos casos do item (i), a substituição de peças poderá ser realizada mediante orçamento prévio aprovado pelo gestor, limitado a [**preencher: % do valor contratual ou nº de ocorrências/ano**].

*(A mesma lógica se aplica ao requisito 2, quanto às estações.)*

---

**A-06 — Seção 3, requisitos 3, 4 e 6 (L83, L84, L90–98)**

*Requisito 6, nova redação do caput (L90):*
> **Qualificação da equipe técnica alocada à execução** (requisito de execução contratual, **não** de habilitação, em atenção à Súmula TCU nº 272), a ser comprovada pela contratada em até 2 (dois) dias úteis antes da assinatura do contrato, **admitidas certificações equivalentes** de conteúdo e carga horária compatíveis, e com substituição de profissional por outro de qualificação igual ou superior, mediante comunicação prévia à fiscalização. A quantidade de profissionais corresponde ao dimensionamento mínimo de [**memória: ex. N1 = x técnicos para 2.148 dispositivos; N2 = y; formação = z**]:

*Lista (L91–98):* manter apenas as certificações cuja **pertinência ao serviço** a equipe justificar em uma linha cada. Recomendações:
- **Workspace Administrator:** pertinente (administração do Admin Console). Manter.
- **Professional Cloud Architect / Associate Cloud Engineer / Cloud Digital Leader:** pertinentes **apenas** ao item 3 (plataforma). Justificar, ou reduzir a 1 profissional.
- **Google Innovator (4x):** credencial de inovação pedagógica docente, cuja concessão depende de seleção do próprio fabricante. **Recomenda-se retirar** ou substituir por "experiência comprovada em formação de estudantes em tecnologia educacional".
- **Google Trainer (4x):** pertinente ao item 5. Justificar a quantidade.

*Requisito 4 (L84):* substituir "classificação 'A' no SecurityScorecard" por "**comprovação de boas práticas de segurança da informação por certificação ISO/IEC 27001 do provedor, ou avaliação independente equivalente**". Substituir "disponibilidade superior a 99,5% (comprovada por órgão independente)" por "disponibilidade mínima mensal de 99,5%, aferida pela contratante a partir de relatório da própria plataforma, com glosa conforme IMR".

*Requisito 5 (L85):* substituir "à unidade da Google Brasil" por "**a empresa ou instituição de referência em tecnologia, preferencialmente a unidade da Google Brasil, condicionada à anuência e à agenda da instituição anfitriã**; na impossibilidade, a contratada indicará local alternativo de natureza equivalente, sujeito à aprovação da fiscalização".

*Seção 5.1 (L123):* retirar "Precedentes semelhantes do Programa Aluno Tutor foram identificados em municípios como Indaiatuba e Itapevi", ou documentá-los com link, número de processo e data.

---

**A-07 — Seção 5.1 (acrescentar após L123)**

> **Alternativa 3 — Atendimento sob demanda (ordem de serviço avulsa por reparo):** menor custo fixo, porém sem SLA, sem preventiva e com o histórico de tempo médio de resolução de 213 dias (Anexo II) no modelo pontual hoje adotado. Custo estimado: [**preencher**].
> **Alternativa 4 — Garantia estendida ou contrato de suporte do fabricante ou revenda dos equipamentos:** cobre apenas defeitos de fabricação, normalmente exclui dano acidental, não abrange estações de carregamento, plataformas ou formação, e não está disponível para os equipamentos de 2021. Custo estimado: [**preencher ou registrar pesquisa negativa**].
> **Alternativa 5 — Renovação do parque em substituição à manutenção:** inviável economicamente diante de [**preencher: custo unitário de reposição × parque**], além de não atender as estações, as plataformas e a formação.
> **Quadro comparativo:** [custo no ciclo de vida de 48 meses × SLA × cobertura × riscos], concluindo pela Alternativa 2.

---

**A-08 — Seção 13 (acrescentar após L220)**

> A vigência plurianual de 48 meses é a mais vantajosa (art. 106, I) porque: (i) dilui os custos de mobilização, implantação do Service Desk, inventário inicial e integração das plataformas; (ii) permite o planejamento de estoque de peças, com reflexo esperado no preço; (iii) coincide com 4 ciclos anuais completos do Programa Aluno Tutor; e (iv) [**preencher: comparação de preço anual × plurianual nas cotações, se obtida**]. A autoridade competente atestará essa vantagem antes da autorização da contratação.

---

**A-09 — Seção 3, requisito 3 (acrescentar ao final da L83)**

> Verificou-se que a edição do Google Workspace for Education atualmente licenciada pelo Município é [**preencher: Fundamentals / Standard / Teaching and Learning / Plus**], cujos recursos nativos de relatórios e exportação [**preencher: incluem / não incluem**] as funcionalidades acima. A plataforma contratada limita-se às funcionalidades **não** disponíveis na edição licenciada, sem sobreposição de pagamento. Ao término do contrato, a contratada entregará os dados e indicadores produzidos em formato aberto e interoperável, e eliminará as cópias sob sua guarda (LGPD, art. 16).

---

**A-10 — Seção 9, item 3 (L181)**

*Original:* "**Habilitação Técnica:** Exigência das certificações individuais da equipe e atestado(s) de capacidade técnica em até **02 dias úteis** antes da assinatura contratual."

*Nova redação:*
> **Habilitação técnica e requisitos de execução:** (a) os atestados de capacidade técnica-operacional (Seção 3, requisito 7) serão exigidos **na fase de habilitação** do certame, limitados às parcelas de maior relevância e a quantitativos mínimos compatíveis com o art. 67 da Lei nº 14.133/2021; (b) as certificações individuais da equipe (Seção 3, requisito 6) serão exigidas **somente do adjudicatário**, em até 02 (dois) dias úteis antes da assinatura do contrato, como requisito de execução (Súmula TCU nº 272).

---

**A-11 — Citações não confirmadas**
- L145: ver C-01.
- L171: manter "Portaria MEC nº 269/2026" **somente se** confirmada no DOU. Caso contrário, usar: "Infraestrutura Nacional de Dados da Educação (Inde/EducaDados), instituída pela Lei Complementar nº 220/2025 e regulamentada pelo Ministério da Educação".
- L134: confirmar que o art. 14, § 3º, do Decreto 4.345/2023 trata da pesquisa de preços. Se não tratar, corrigir a remissão.
- L290: juntar o link do PNCP, a data e a natureza do valor da referência do TRT-16.

---

**M-12 — Seção 1 (L26 e L44)**
> *L26:* "[...] a rede municipal alcançou o **primeiro lugar da Baixada Santista no Ideb dos anos iniciais do ensino fundamental** nas edições [**preencher: 2023 e 2025**], com nota [**preencher conforme Inep**] em [**edição**], superando a média nacional (fonte: Inep, consulta em [**data**])."

Uniformizar a L44 com a mesma fonte e eliminar a coexistência de "6,38" e "6,3" para a mesma edição.

**M-13 — Seção 2 (L75):** acrescentar "A contratação consta do Plano de Contratações Anual de [**ano**], item/código [**preencher**], conforme [**ato/publicação**]". Se não constar, incluir a justificativa da ausência e a providência de inclusão.

**M-14 — Seção 10 (acrescentar linhas):**
- **6. Descontinuidade orçamentária:** Prob. Média | Imp. Alta | Ação preventiva: atestado anual de créditos (art. 106, II) e previsão no PPA | Contingência: extinção sem ônus (art. 106, III) ou supressão de itens (Resp.: Gestor/Secretaria).
- **7. Impugnação por restritividade:** Prob. Média | Imp. Alta | Ação preventiva: justificar cada requisito e aceitar equivalentes (A-06) | Contingência: revisar o edital e republicar.
- **8. Pagamento em duplicidade com garantias:** Prob. Média | Imp. Média | Ação preventiva: quadro de garantias (A-04) | Contingência: glosa.
- **9. Indisponibilidade do local da visita:** Prob. Média | Imp. Baixa | Ação preventiva: cláusula de local alternativo | Contingência: substituição aprovada pela fiscalização.
- **10. Dependência tecnológica (lock-in) e perda de dados ao término:** Prob. Média | Imp. Alta | Ação preventiva: cláusula de portabilidade e transição | Contingência: plano de transição contratual.
- **11. Tratamento de dados de crianças e adolescentes:** Prob. Baixa | Imp. Alta | Ação preventiva: base legal (LGPD, art. 14), termo de consentimento e acordo de tratamento de dados com a contratada | Contingência: suspensão do tratamento e comunicação à ANPD.

*Nota L196:* corrigir para "O Risco 6 trata da descontinuidade orçamentária (art. 106, II e III)".

**M-15 — Seção 7 (L157–159):** definir **Crítica** como indisponibilidade que afete ≥ [x]% dos equipamentos de uma unidade escolar ou falha da plataforma ou do MDM; **Alta** como equipamento individual inoperante; **Média** como falha parcial; **Baixa** como dúvida ou configuração. Vincular a um Instrumento de Medição de Resultado (IMR) com glosas no TR. *Seção 8:* acrescentar indicadores com linha de base do Anexo II, por exemplo: "reduzir o percentual de equipamentos inoperantes de 80,4% (base: registros de 2026) para ≤ [x]%"; "tempo médio de reparo de 213 dias para ≤ 15 dias úteis"; "120 alunos certificados".

**M-16 — Seção 13 (L220):** "[...] reajuste anual pelo **IPCA/IBGE** [ou ICTI/IPEA, a critério da equipe, com justificativa], com data-base vinculada à data do orçamento estimado (art. 25, § 7º, da Lei nº 14.133/2021)".

**M-17 — Seção 1 (L66):** "A não disponibilidade de créditos orçamentários em exercícios futuros, verificada no atestado anual previsto no art. 106, II, da Lei nº 14.133/2021, poderá ensejar a extinção do contrato sem ônus (art. 106, III), acarretando: [...]"

**M-19 — Seção 3(5)(iii) (L88):** "Cobertura por **seguro de acidentes pessoais coletivo** em favor dos alunos e acompanhantes durante a atividade, mantido pela contratada". *Anexo V:* (a) uniformizar o acompanhamento no trajeto: "servidor(es) da frota municipal **e educador(es) designado(s) da rede municipal**"; (b) acrescentar a cláusula: "Autorizo o tratamento dos dados pessoais do(a) aluno(a) estritamente para a finalidade desta atividade, nos termos do art. 14 da Lei nº 13.709/2018, e [**autorizo / não autorizo**] o uso de imagem em comunicações institucionais".

**M-20 — Anexo III (L289):** listar F1 a F17 em matriz com: órgão, número do processo, data, link do PNCP, objeto, unidade, valor, natureza do valor (estimado, adjudicado, contratado ou registrado em ata), classificação 🟢/🟡/🔴 e motivo.

**M-21 — Seção 13 (L219):** "Os serviços enquadram-se como **comuns** (art. 6º, XIII), pois seus padrões de desempenho e qualidade, inclusive os do Programa Aluno Tutor (carga horária, eixos, produtos e certificação), são objetivamente definidos no Termo de Referência por especificações usuais de mercado."

**B-18 — Anexo II, item 3:** Tela 15 (33,3%); Teclado 14 (31,1%); Sistema 8 (17,8%); Bateria 6 (13,3%); Outros 2 (4,4%), sobre n = 45.

**B-22 — Introdução:** inserir após a L10 o texto-padrão do modelo (D2, L12–14).

---

## PROVIDÊNCIAS MANUAIS (não classificadas como falhas)

- Assinatura e data do ETP e dos anexos; composição completa da equipe de planejamento (Anexo I).
- Portaria de designação de gestor e fiscal.
- Dotação orçamentária, item do PCA e datas de inventário.
- Conferir no sítio oficial a redação vigente do **Decreto 4.345/2023** (arts. 8º e 14) e do **Guia de Orientação municipal (2025)**: **inacessíveis a esta auditoria**.
- Conferir no portal do TCU o inteiro teor do **Acórdão 3.569/2023-2ª Câmara**; no DOU, a **Portaria MEC 269/2026**; no Inep, o **Ideb** de Bertioga.
- Obter dos fornecedores o esclarecimento escrito sobre a unidade dos valores R$ 198,00 e R$ 475,00.
- Executar as buscas no PNCP e no Painel de Preços do roteiro do Agente 4 e salvar as telas.

## FONTES CONSULTADAS (16/11/2026)

- Lei 14.133/2021: [Planalto](https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm) (localizada; texto integral não aberto); D2 (transcrição do art. 18 e do art. 23)
- Decreto 4.345/2023: [Bertioga](https://www.bertioga.sp.gov.br/wp/wp-content/uploads/2024/01/DECRETO-4345-27-DE-DEZEMBRO-DE-2023-REGULAMENTACAO-LEI-14133-DE-2021.pdf) (**inacessível**)
- Art. 106: [Justen](https://justen.com.br/wp-content/uploads/2023/06/IE182PrazoContratualLei14133.pdf); [Zênite](https://zenite.blog.br/nova-lei-de-licitacoes-prazo-inicial-e-prorrogacao-dos-contratos-de-servicos-continuos/)
- Súmula 247: [TCU – Licitações e Contratos](https://licitacoesecontratos.tcu.gov.br/4-1-8-justificativas-para-o-parcelamento-ou-nao-da-contratacao); [TCDF](https://jurisprudencia.tc.df.gov.br/wp-content/uploads/2018/07/Objeto-divisível.-Adjudicação-por-item.-Súmula-TCU-247.pdf)
- Súmula 272: [TCDF](https://jurisprudencia.tc.df.gov.br/wp-content/uploads/2018/07/Custos-antecipados-para-as-licitantes.-Súmula-TCU-272.pdf); [Zênite](https://zenite.blog.br/topico/credenciamento-pelo-fabricante/)
- Acórdão 3.569/2023-2ªC: [Zênite](https://zenite.blog.br/vicios-no-planejamento/) (secundária)
- LC 220/2025: [Câmara dos Deputados](https://www2.camara.leg.br/legin/fed/leicom/2025/leicomplementar-220-31-outubro-2025-798216-norma-pl.html)
- EducaDados: [Agência Gov](https://agenciagov.ebc.com.br/noticias/202603/startup-gov-br-gestao-e-mec-lancam-plataforma-nacional-de-dados-da-educacao); [Congresso em Foco](https://www.congressoemfoco.com.br/artigo/117658/governanca-de-dados-para-qualidade-da-gestao-e-equidade-educacional)
- Ideb: [Diário do Litoral](https://www.diariodolitoral.com.br/bertioga/bertioga-tornou-se-a-cidade-da-oportunidade-e-do/208655/); [JB](https://www.jb.com.br/brasil/educacao/2026/08/1060517-ideb-2025-registra-maior-evolucao-acumulada-em-20-anos.html)
- PNCP (API): **acesso bloqueado** no ambiente

---

## DECISÃO SOLICITADA (FASE 5)

Indique, por ID, quais alterações aprovar (por exemplo, "aprovo C-01, C-02, A-04 a A-11 e M-12 a M-21; A-03 pela Opção A; B-23 sem alteração"). Os campos marcados com **[preencher]** dependem de dados que só a Secretaria possui e **não serão inventados**: ficarão sinalizados na versão revisada (v2), caso não sejam informados.
