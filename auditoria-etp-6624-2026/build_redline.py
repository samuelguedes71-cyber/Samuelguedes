"""Gera a minuta marcada (v2) do ETP a partir do v1, sem alterar o original.
Marcadores: {+...+} = inserção (marca-texto) ; {-...-} = exclusão (vermelho tachado)."""
import re, sys

SRC, OUT = sys.argv[1], sys.argv[2]
t = open(SRC, encoding="utf-8").read()
log = []

def I(s): return "{+" + s + "+}"
def D(s):
    s = s.replace(r"$8\times5$", "8×5").replace(r"$\times$", "×")
    return "{-" + s + "-}"

def rep(old, new, tag):
    global t
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"[{tag}] trecho encontrado {n} vezes: {old[:80]!r}")
    t = t.replace(old, new)
    log.append(tag)

def wrap_row(row, f):
    cells = row.strip().strip("|").split("|")
    return "| " + " | ".join(f(c.strip()) if c.strip() else "" for c in cells) + " |"

def wrap_table(tbl, f):
    out = []
    for r in tbl.strip("\n").split("\n"):
        out.append(r if re.match(r"^\|\s*:?-", r) else wrap_row(r, f))
    return "\n".join(out)

P = "⟦"; Q = "⟧"
def ph(s): return P + s + Q

# ---------- Nota de versão + Introdução (B-22)
rep("**Secretaria Municipal de Educação**\n",
    "**Secretaria Municipal de Educação**\n\n> " + I("**VERSÃO DE REVISÃO (v2 — minuta marcada).** Legenda: texto com marca-texto amarelo = redação nova proposta; texto em vermelho tachado = trecho a excluir. Campos entre " + ph("…") + " dependem de dados da Secretaria. A fundamentação de cada alteração consta do Relatório de Diagnóstico e do Adendo nº 1 da auditoria de 16/11/2026.") + "\n", "nota-versao")
rep("**Base Legal:** Lei Federal nº 14.133/2021 (art. 18)\n",
    "**Base Legal:** Lei Federal nº 14.133/2021 (art. 18)\n\n## " + I("INTRODUÇÃO") + "\n\n" +
    I("O presente documento caracteriza a primeira etapa da fase de planejamento e apresenta os devidos estudos para a contratação de solução que atenderá à necessidade abaixo especificada.") + "\n\n" +
    I("O objetivo principal é estudar detalhadamente a necessidade e identificar no mercado a melhor solução para supri-la, em observância às normas vigentes e aos princípios que regem a Administração Pública.") + "\n", "B-22")

# ---------- Seção 1
rep("* No campo dos resultados de aprendizagem, a rede municipal alcançou, pelo segundo ano consecutivo, o **primeiro lugar da Baixada Santista no Índice de Desenvolvimento da Educação Básica (Ideb)** nos anos iniciais do ensino fundamental (nota 6,38, resultado 2024), superando a média nacional.",
    "* No campo dos resultados de aprendizagem, a rede municipal alcançou" + D(", pelo segundo ano consecutivo,") + " o **primeiro lugar da Baixada Santista no Índice de Desenvolvimento da Educação Básica (Ideb)** nos anos iniciais do ensino fundamental " + D("(nota 6,38, resultado 2024)") + I("nas edições " + ph("preencher: 2023 e 2025") + ", com nota " + ph("preencher conforme Inep") + " em " + ph("edição")) + ", superando a média nacional" + I(" (fonte: Inep, consulta em " + ph("data") + ")") + ".", "M-12a")
rep("com nota **6,6** (ante 6,3 em 2023).",
    "com nota **6,6** (ante 6,3 em 2023)" + I(" " + ph("conferir com o Inep e uniformizar com a informação do item anterior")) + ".", "M-12b")
rep("A não aprovação dos créditos orçamentários em exercícios futuros (art. 106, inciso II, da Lei nº 14.133/2021) acarretaria:",
    D("A não aprovação dos créditos orçamentários em exercícios futuros (art. 106, inciso II, da Lei nº 14.133/2021) acarretaria:") + " " + I("A não disponibilidade de créditos orçamentários em exercícios futuros, verificada no atestado anual previsto no art. 106, II, da Lei nº 14.133/2021, poderá ensejar a extinção do contrato sem ônus (art. 106, III), acarretando:"), "M-17")

# ---------- Seção 2
rep("assegurando compatibilidade orçamentária e previsibilidade na execução dos recursos destinados à educação digital.",
    "assegurando compatibilidade orçamentária e previsibilidade na execução dos recursos destinados à educação digital." + I(" A contratação consta do Plano de Contratações Anual de " + ph("ano") + ", item/código " + ph("preencher") + ", conforme " + ph("ato/publicação") + ". " + ph("Caso não conste: justificar a ausência e indicar a providência de inclusão.")), "M-13")

# ---------- Seção 3
rep("para os Chromebooks, sem cobrança adicional.",
    "para os Chromebooks, sem cobrança adicional." + I(" Não se incluem na cobertura sem custo: (i) danos decorrentes de vandalismo, uso indevido comprovado em laudo, sinistro (incêndio, inundação, furto) ou força maior; e (ii) equipamentos declarados irrecuperáveis em laudo técnico circunstanciado, aprovado pela fiscalização, os quais serão baixados da base de cobertura com o correspondente ajuste do quantitativo faturado. Nos casos do item (i), a substituição de peças poderá ser realizada mediante orçamento prévio aprovado pelo gestor, limitado a " + ph("preencher: % do valor contratual ou nº de ocorrências/ano") + "."), "A-05a")
rep("peças inclusas (rodas, fechaduras, contatos elétricos).",
    "peças inclusas (rodas, fechaduras, contatos elétricos)." + I(" Aplicam-se às estações as mesmas exclusões de cobertura e a mesma regra de laudo de irrecuperabilidade previstas no requisito 1."), "A-05b")
rep("suporte a até **9.920 acessos simultâneos**,",
    "suporte a até **9.920 acessos simultâneos**" + I(" (memória de cálculo: " + ph("nº de alunos + docentes + gestores com conta institucional") + ")") + ",", "A-09a")
rep("Consolida a administração dos Chromebooks via Google Admin Console e entrega indicadores qualitativos de uso por unidade escolar e aplicativo.",
    "Consolida a administração dos Chromebooks via Google Admin Console e entrega indicadores qualitativos de uso por unidade escolar e aplicativo." +
    I(" Verificou-se que a edição do Google Workspace for Education atualmente licenciada pelo Município é " + ph("preencher: Fundamentals / Standard / Teaching and Learning / Plus") + ", cujos recursos nativos de relatórios e exportação " + ph("incluem / não incluem") + " as funcionalidades acima; a plataforma contratada limita-se às funcionalidades não disponíveis na edição licenciada, sem sobreposição de pagamento (art. 24 do Decreto Municipal nº 4.345/2023). Ao término do contrato, a contratada entregará os dados e indicadores produzidos em formato aberto e interoperável e eliminará as cópias sob sua guarda (art. 16 da Lei nº 13.709/2018).") +
    I(" A plataforma deverá observar os padrões técnicos, os modelos informacionais e os requisitos de segurança da Infraestrutura Nacional de Dados da Educação (EducaDados), à medida que forem editados pelo Ministério da Educação (arts. 9º a 11 e 22 da Portaria MEC nº 269/2026), e, quando houver tratamento de dados em nível de estudante, adotará o CPF como Identificador Nacional Único do Estudante (art. 7º da mesma Portaria), observados a Lei nº 13.709/2018, em especial o art. 14, e os princípios da necessidade e da adequação."), "A-09b/M-24a")
rep('Sistema em nuvem com disponibilidade superior a **99,5%** (comprovada por órgão independente) e classificação **"A" no SecurityScorecard**, com geolocalização e restrição de aplicativos.',
    "Sistema em nuvem com disponibilidade " + D('superior a **99,5%** (comprovada por órgão independente) e classificação **"A" no SecurityScorecard**') + " " + I("mínima mensal de **99,5%**, aferida pela contratante a partir de relatório da própria plataforma, com glosa conforme IMR, e comprovação de boas práticas de segurança da informação por certificação ISO/IEC 27001 do provedor ou avaliação independente equivalente") + ", com geolocalização e restrição de aplicativos.", "A-06a")
rep("Inclui visita técnico-pedagógica presencial de um dia (sem pernoite) à unidade da Google Brasil, condicionada a:",
    "Inclui visita técnico-pedagógica presencial de um dia (sem pernoite) " + D("à unidade da Google Brasil, condicionada a:") + " " + I("a empresa ou instituição de referência em tecnologia, preferencialmente a unidade da Google Brasil, condicionada à anuência e à agenda da instituição anfitriã; na impossibilidade, a contratada indicará local alternativo de natureza equivalente, sujeito à aprovação da fiscalização. A visita fica, ainda, condicionada a:"), "A-06b")
rep("Cobertura por seguro de acidentes a terceiros mantido pela contratada;",
    "Cobertura por seguro de " + D("acidentes a terceiros") + " " + I("acidentes pessoais coletivo em favor dos alunos e acompanhantes durante a atividade,") + " mantido pela contratada;", "M-19a")
rep("6. **Certificações e Qualificações Técnicas da Contratada** *(prazo: 02 dias úteis antes da assinatura do contrato)*:",
    "6. " + D("**Certificações e Qualificações Técnicas da Contratada** *(prazo: 02 dias úteis antes da assinatura do contrato)*:") + " " + I("**Qualificação da equipe técnica alocada à execução** (requisito de execução contratual, não de habilitação, em atenção à Súmula TCU nº 272), a ser comprovada pela contratada em até 02 (dois) dias úteis antes da assinatura do contrato, admitidas certificações equivalentes de conteúdo e carga horária compatíveis, e com substituição de profissional por outro de qualificação igual ou superior, mediante comunicação prévia à fiscalização. A quantidade de profissionais corresponde ao dimensionamento mínimo de " + ph("memória: N1 = x técnicos para 2.148 dispositivos; N2 = y; formação = z") + ":"), "A-06c")
rep("   * 02x Google Cloud Certified Professional Cloud Architect;", "   * 02x Google Cloud Certified Professional Cloud Architect;" + I(" *(pertinente ao item 3; " + ph("justificar a quantidade ou reduzir a 1") + ")*"), "A-06d")
rep("   * 02x Cloud Digital Leader;", "   * 02x Cloud Digital Leader;" + I(" *(pertinente ao item 3; " + ph("justificar a quantidade ou reduzir a 1") + ")*"), "A-06e")
rep("   * 01x Google Cloud Certified Associate Cloud Engineer;", "   * 01x Google Cloud Certified Associate Cloud Engineer;" + I(" *(pertinente ao item 3)*"), "A-06f")
rep("   * 04x Google Trainer;", "   * 04x Google Trainer;" + I(" *(pertinente ao item 5; " + ph("justificar a quantidade") + ")*"), "A-06g")
rep("   * 04x Google Innovator.", "   * " + D("04x Google Innovator.") + " " + I(ph("quantidade") + " profissional(is) com experiência comprovada em formação de estudantes em tecnologia educacional."), "A-06h")
rep("*Dispensa-se carta de credenciamento ou certificado de parceria comercial pelo fabricante, em atenção à Súmula TCU nº 272.*",
    "*Dispensa-se carta de credenciamento ou certificado de parceria comercial pelo fabricante, em atenção à Súmula TCU nº 272.*" + I(" Admite-se, nos termos do art. 28 do Decreto Municipal nº 4.345/2023, a substituição dos atestados por termo de contrato ou notas fiscais que comprovem a execução de objeto compatível, mediante diligência do Agente de Contratação."), "A-10a")

# ---------- Seção 4 (C-02)
old_tbl = """| Item | Descrição do Serviço | Unidade | Quantidade | Vigência |
| :--- | :--- | :--- | :---: | :---: |
| **1** | Suporte técnico e manutenção de Chromebooks (NOC $8\\times5$) | Dispositivo | 2.148 | 48 meses |
| **2** | Suporte e manutenção do sistema de carregamento de estações (59 unid., sendo 20 Laboratórios Multimídia — periféricos excluídos) | Unidade | 59 | 48 meses |
| **3** | Plataforma de governança de dados educacionais com acesso seguro (inclui gestão dos Chromebooks via Admin Console) | Mês | 1 | 48 meses |
| **4** | Solução de Gerenciamento de Dispositivos Móveis (MDM) exclusiva para os 40 tablets | Mês (40 licenças) | 40 | 48 meses |
| **5** | Programa Aluno Tutor de Tecnologia (30 alunos/ano = 120 alunos totais) | Mês | 30 | 48 meses |"""
new_tbl = """| Item | Descrição do Serviço | Unidade de medida | Quantidade | Período | Fórmula do total |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | Suporte técnico NOC 8×5 e manutenção de Chromebooks, com peças | dispositivo/mês | 2.148 | 48 meses | P1 × 2.148 × 48 |
| **2** | Suporte e manutenção preventiva do sistema de carregamento, com peças (periféricos multimídia excluídos) | estação/mês | 59 | 48 meses | P2 × 59 × 48 |
| **3** | Plataforma de governança de dados educacionais com acesso seguro | serviço/mês | 1 | 48 meses | P3 × 1 × 48 |
| **4** | Solução MDM exclusiva para os 40 tablets | licença/mês | 40 | 48 meses | P4 × 40 × 48 |
| **5** | Programa Aluno Tutor de Tecnologia | aluno/ano | 30 por ano (120 no total) | 4 ciclos anuais | P5 × 30 × 4 |"""
mem = [
    "**Memória de cálculo.**",
    "(a) **Item 1:** 2.148 Chromebooks do inventário patrimonial em " + ph("data") + ". Os equipamentos ainda cobertos por garantia contratual do fornecedor serão incluídos na cobertura apenas a partir do término da respectiva garantia, conforme o quadro de garantias por lote (Seção 11, item 3).",
    "(b) **Item 2:** 59 estações (20 do PP 87/2021 e 39 de " + ph("origem a preencher") + ").",
    "(c) **Item 3:** 1 ambiente; o limite de 9.920 acessos simultâneos corresponde a " + ph("memória: nº de alunos + docentes + gestores com conta institucional") + ".",
    "(d) **Item 4:** 40 tablets incompatíveis com o Google Admin Console (inventário em " + ph("data") + ").",
    "(e) **Item 5:** 30 alunos por ciclo anual — dimensionamento conservador em relação à média histórica de 40 alunos/ano —, com 4 ciclos na vigência de 48 meses, totalizando 120 alunos. O preço P5 é por aluno e por ciclo anual completo (90 horas, materiais, alimentação, formatura e seguro).",
]
rep(old_tbl, wrap_table(old_tbl, D) + "\n\n" + I("*Quadro substituído — unidades de medida padronizadas:*") + "\n\n" + wrap_table(new_tbl, I) + "\n\n" + "\n\n".join(I(m) for m in mem), "C-02a")

# ---------- Seção 5 (A-03 Opção A, A-07)
rep("## 5. LEVANTAMENTO DE MERCADO E JUSTIFICATIVA DO LOTE ÚNICO",
    "## 5. LEVANTAMENTO DE MERCADO E JUSTIFICATIVA " + D("DO LOTE ÚNICO") + " " + I("DO PARCELAMENTO"), "A-03a")
rep(" Precedentes semelhantes do Programa Aluno Tutor foram identificados em municípios como Indaiatuba e Itapevi.",
    " " + D("Precedentes semelhantes do Programa Aluno Tutor foram identificados em municípios como Indaiatuba e Itapevi.") + "\n" +
    "* " + I("**Alternativa 3 — Atendimento sob demanda (ordem de serviço avulsa por reparo):** menor custo fixo, porém sem SLA, sem manutenção preventiva e com o histórico de tempo médio de resolução de 213 dias (Anexo II) no modelo pontual hoje adotado. Custo estimado: " + ph("preencher") + ".") + "\n" +
    "* " + I("**Alternativa 4 — Garantia estendida ou contrato de suporte do fabricante ou revenda dos equipamentos:** cobre apenas defeitos de fabricação, normalmente exclui dano acidental, não abrange estações de carregamento, plataformas ou formação, e não está disponível para os equipamentos de 2021. Custo estimado: " + ph("preencher ou registrar pesquisa negativa") + ".") + "\n" +
    "* " + I("**Alternativa 5 — Renovação do parque em substituição à manutenção:** inviável economicamente diante de " + ph("preencher: custo unitário de reposição × parque") + ", além de não atender as estações, as plataformas e a formação.") + "\n" +
    "* " + I("**Quadro comparativo:** " + ph("custo no ciclo de vida de 48 meses × SLA × cobertura × riscos de cada alternativa") + ", concluindo pela Alternativa 2, conforme o art. 22, § 1º, do Decreto Municipal nº 4.345/2023 (consideração do ciclo de vida do objeto)."), "A-07")
rep("### 5.2. Justificativa de Não Parcelamento (Lote Único)",
    "### 5.2. Justificativa " + D("de Não Parcelamento (Lote Único)") + " " + I("do Parcelamento (Grupo 1 e Itens Autônomos)"), "A-03b")
old52 = """Em consonância com a **Súmula TCU nº 247**, opta-se pelo Lote Único por:
* **I. Viabilidade técnica e econômica:** Interdependência operacional entre os itens 1 e 3 (gestão via Google Admin Console), compartilhamento de logística bimestral entre os itens 1 e 2, e aproveitamento da mesma equipe certificada Google para o item 5.
* **II. Ausência de perda de escala:** O parcelamento replicaria estruturas de Service Desk e SLAs, elevando custos administrativos e reduzindo o poder de negociação por volume."""
lines52 = old52.split("\n")
del52 = D(lines52[0]) + "\n" + "\n".join("* " + D(l[2:]) for l in lines52[1:])
new52 = "\n\n" + I('Em observância ao princípio do parcelamento (art. 47, II, e art. 18, § 1º, VIII, da Lei nº 14.133/2021; art. 5º, § 1º, VI, "b", do Decreto Municipal nº 4.345/2023) e à Súmula TCU nº 247, a licitação será organizada em:') + "\n\n" + \
    "* " + I("**Grupo 1 — Itens 1, 2 e 3**, adjudicados em conjunto em razão da interdependência operacional: administração dos Chromebooks no Google Admin Console integrada à plataforma de governança, Service Desk único e logística comum das visitas bimestrais aos mesmos equipamentos e estações;") + "\n" + \
    "* " + I("**Item 4 — Solução MDM**, adjudicação autônoma, por existir mercado próprio de fornecedores de gerenciamento de dispositivos móveis;") + "\n" + \
    "* " + I("**Item 5 — Programa Aluno Tutor de Tecnologia**, adjudicação autônoma, por se tratar de serviço de formação com mercado próprio.") + "\n\n" + \
    I("O agrupamento dos itens 1, 2 e 3 evita a replicação de estruturas de Service Desk e de SLAs e preserva a economia de escala; a separação dos itens 4 e 5 amplia a competitividade sem prejuízo ao conjunto da solução.")
rep(old52, del52 + new52, "A-03c")

# ---------- Seção 6 (C-01, C-02, A-11a)
old6 = "A estimativa observou o art. 23, § 1º, da Lei nº 14.133/2021 e o art. 14, § 3º, do Decreto Municipal nº 4.345/2023. Como a pesquisa em fontes públicas abertas (PNCP e Painel de Preços) não localizou contratações tecnicamente comparáveis para os cinco itens, aplica-se a **cotação direta com no mínimo 3 fornecedores especializados** (art. 23, § 1º, IV, da Lei nº 14.133/2021), exigindo decomposição de custos (peças nos itens 1 e 2; custos por turma/aluno no item 5)."
new6 = D(old6) + "\n\n" + \
    I("**6.1. Natureza da estimativa.** A estimativa apresentada nesta seção tem caráter preliminar, destina-se à avaliação da viabilidade econômica da solução escolhida (art. 18, § 1º, VI, da Lei nº 14.133/2021) e não se confunde com o valor de referência da licitação, que será fixado no Termo de Referência a partir da pesquisa de preços formal de que trata o art. 23 da mesma Lei, documentada no Anexo III.") + "\n\n" + \
    I("**6.2. Fontes e método.** A pesquisa em bancos públicos (PNCP e Painel de Preços), priorizada nos termos do art. 14, § 3º, do Decreto Municipal nº 4.345/2023 e documentada no Anexo III, não localizou contratações tecnicamente comparáveis para os cinco itens. A estimativa preliminar baseia-se em " + ph("preencher: nº de orçamentos preliminares, fornecedores e datas") + ", normalizados às unidades da Seção 4, e será substituída pela cotação formal com no mínimo 3 (três) fornecedores (art. 23, § 1º, IV, da Lei nº 14.133/2021; art. 14, § 4º, do Decreto Municipal nº 4.345/2023), acompanhada da justificativa de escolha de cada fornecedor e da decomposição de custos (peças nos itens 1 e 2; custos por aluno no item 5).") + "\n\n" + \
    I("**6.3. Quadro de estimativa preliminar.**")
rep(old6, new6, "C-01a")
rep("| Item | Descrição do Serviço | Unid. | Qtd. | Valor Unit. Mensal | Valor Total Mensal | Valor Total (48 meses) |",
    "| Item | Descrição do Serviço | Unid. | Qtd. | " + D("Valor Unit. Mensal") + " " + I("Preço Unitário (na unidade)") + " | " + D("Valor Total Mensal") + " " + I("Valor Total Anual") + " | Valor Total (48 meses) |", "C-02b")
rep("| **1** | Suporte e manutenção de Chromebooks | Disp. | 2.148 |", "| **1** | Suporte e manutenção de Chromebooks | Disp." + I("/mês") + " | 2.148 |", "C-02c")
rep("| **2** | Suporte e manutenção de estações de carregamento | Unid. | 59 |", "| **2** | Suporte e manutenção de estações de carregamento | " + D("Unid.") + " " + I("Estação/mês") + " | 59 |", "C-02d")
rep("| **3** | Plataforma de governança de dados educacionais | Mês | 1 |", "| **3** | Plataforma de governança de dados educacionais | " + D("Mês") + " " + I("Serviço/mês") + " | 1 |", "C-02e")
rep("| **4** | Solução MDM (40 tablets) | Mês | 40 |", "| **4** | Solução MDM (40 tablets) | " + D("Mês") + " " + I("Licença/mês") + " | 40 |", "C-02f")
rep("| **5** | Programa Aluno Tutor de Tecnologia | Mês | 30 |", "| **5** | Programa Aluno Tutor de Tecnologia | " + D("Mês") + " " + I("Aluno/ano") + " | 30" + I(" (× 4 ciclos)") + " |", "C-02g")
old_adv = "> **ADVERTÊNCIA:** Os valores preliminares informais não devem fundamentar a publicação do edital (Acórdão TCU nº 3.569/2023 – 2ª Câmara). A viabilidade econômica definitiva está condicionada à conclusão do mapa de preços formal documentado no Relatório de Pesquisa de Preços (Anexo III)."
new_adv = I("Fórmulas: itens 1 a 4 — preço unitário × quantidade × 48 meses; item 5 — preço por aluno/ciclo anual × 30 × 4 ciclos. Valores: " + ph("preencher a partir das propostas preliminares, normalizadas às unidades acima") + ".") + "\n\n" + \
    I("**6.4. Disponibilidade orçamentária.** A despesa estimada para o exercício de " + ph("ano") + " correrá à conta da dotação " + ph("preencher") + ". Para os exercícios seguintes, a continuidade depende de previsão no Plano Plurianual e nas leis orçamentárias, com atestado anual da existência de créditos (art. 106, II, da Lei nº 14.133/2021).") + "\n\n" + \
    "> " + D("**ADVERTÊNCIA:** Os valores preliminares informais não devem fundamentar a publicação do edital (Acórdão TCU nº 3.569/2023 – 2ª Câmara). A viabilidade econômica definitiva está condicionada à conclusão do mapa de preços formal documentado no Relatório de Pesquisa de Preços (Anexo III).") + "\n\n" + \
    "> " + I("**ADVERTÊNCIA METODOLÓGICA:** A cotação direta com fornecedores (art. 23, § 1º, IV, da Lei nº 14.133/2021) é adotada de forma subsidiária, depois da busca documentada nos parâmetros dos incisos I e II do mesmo dispositivo, cuja priorização é determinada pelo art. 14, § 3º, do Decreto Municipal nº 4.345/2023. O Tribunal de Contas da União, no Acórdão nº 3.569/2023 – 2ª Câmara, considerou irregular a pesquisa de preços realizada exclusivamente com potenciais fornecedores, sem considerar contratações similares da Administração, por ter propiciado sobrepreço substancial no orçamento estimado. Por essa razão, integram o Anexo III as telas de consulta às fontes públicas, inclusive as de resultado negativo, e os orçamentos obtidos serão analisados criticamente, com desconsideração motivada dos valores inexequíveis, inconsistentes ou excessivamente elevados (art. 14, caput e §§ 5º e 6º, do Decreto Municipal nº 4.345/2023). Os valores preliminares informais servem apenas à avaliação de viabilidade desta etapa e serão substituídos pelo mapa de preços formal no Termo de Referência.")
rep(old_adv, new_adv, "C-01b/A-11a")

# ---------- Seção 7 (M-15, A-06 coerência)
rep("e culminância com formatura e visita presencial de 1 dia à sede da Google Brasil.",
    "e culminância com formatura e visita presencial de 1 dia " + D("à sede da Google Brasil") + " " + I("a empresa ou instituição de referência em tecnologia, preferencialmente a Google Brasil (Seção 3, requisito 5)") + ".", "A-06i")
rep("  * *Severidade Alta (reparo físico fora de garantia):* Resolução/retorno do equipamento em até 15 dias úteis, com peças inclusas.",
    "  * *Severidade Alta (reparo físico fora de garantia):* Resolução/retorno do equipamento em até 15 dias úteis, com peças inclusas.\n" +
    "  * " + I("*Definições:* Crítica — indisponibilidade que afete " + ph("x") + "% ou mais dos equipamentos de uma unidade escolar, ou falha da plataforma de governança ou do MDM; Alta — equipamento individual inoperante; Média — falha parcial; Baixa — dúvida ou configuração.") + "\n" +
    "  * " + I("Os níveis de serviço serão vinculados a Instrumento de Medição de Resultado (IMR), com glosas, a ser detalhado no Termo de Referência."), "M-15a")

# ---------- Seção 8 (M-15, M-24, A-06)
rep("(incluindo visita à sede da Google Brasil).",
    "(incluindo visita " + D("à sede da Google Brasil") + " " + I("técnico-pedagógica a empresa ou instituição de referência em tecnologia") + ").", "A-06j")
rep("* **Alinhamento à Política Nacional de Dados Educacionais:** Alinhamento da Plataforma de Governança à Infraestrutura Nacional de Dados da Educação (**EducaDados**, Portaria MEC nº 269/2026 e Lei Complementar nº 220/2025).",
    "* **Alinhamento à Política Nacional de Dados Educacionais:** " + D("Alinhamento da Plataforma de Governança à Infraestrutura Nacional de Dados da Educação (**EducaDados**, Portaria MEC nº 269/2026 e Lei Complementar nº 220/2025).") + " " + I("Compatibilidade da Plataforma de Governança com os padrões da Infraestrutura Nacional de Dados da Educação (**EducaDados**), regulamentada pela Portaria MEC nº 269/2026 com base na Lei Complementar nº 220/2025.") + "\n" +
    "* " + I("**Indicadores de resultado (linha de base: Anexo II):** reduzir o percentual de equipamentos registrados fora de uso de 80,4% para no máximo " + ph("x") + "%; reduzir o tempo médio de resolução de 213 dias para até 15 dias úteis (severidade Alta); certificar 120 alunos no Programa Aluno Tutor ao longo da vigência."), "M-24b/M-15b")

# ---------- Seção 9 (A-10)
rep("3. **Habilitação Técnica:** Exigência das certificações individuais da equipe e atestado(s) de capacidade técnica em até **02 dias úteis** antes da assinatura contratual.",
    "3. " + D("**Habilitação Técnica:** Exigência das certificações individuais da equipe e atestado(s) de capacidade técnica em até **02 dias úteis** antes da assinatura contratual.") + " " + I("**Habilitação técnica e requisitos de execução:** (a) os atestados de capacidade técnico-operacional (Seção 3, requisito 7) serão exigidos na fase de habilitação do certame, limitados às parcelas de maior relevância e a quantitativos mínimos compatíveis com o art. 67 da Lei nº 14.133/2021, admitida a substituição prevista no art. 28 do Decreto Municipal nº 4.345/2023; (b) as certificações individuais da equipe (Seção 3, requisito 6) serão exigidas somente do adjudicatário, em até 02 (dois) dias úteis antes da assinatura do contrato, como requisito de execução (Súmula TCU nº 272)."), "A-10b")
rep("dos alunos do Programa Aluno Tutor até a Google Brasil.",
    "dos alunos do Programa Aluno Tutor até " + D("a Google Brasil") + " " + I("o local da visita técnico-pedagógica") + ".", "A-06k")

# ---------- Seção 10 (M-14)
anchor10 = "| Tratamento anticorrosivo e inspeção nas visitas bimestrais. *(Resp.: Fiscal)* | Substituição antecipada de componentes fora do calendário padrão. *(Resp.: Fiscal)* |"
rows = [
    ["**6. Descontinuidade orçamentária**", "Média", "Alta", "Interrupção dos serviços e do Programa Aluno Tutor.", "Atestado anual de créditos (art. 106, II) e previsão no PPA. *(Resp.: Gestor/Secretaria)*", "Extinção sem ônus (art. 106, III) ou supressão de itens. *(Resp.: Secretaria)*"],
    ["**7. Impugnação por restritividade de requisitos**", "Média", "Alta", "Suspensão ou atraso do certame.", "Justificativa de pertinência de cada requisito e aceitação de equivalentes. *(Resp.: Gestor)*", "Revisão do edital e republicação. *(Resp.: Gestor)*"],
    ["**8. Pagamento em duplicidade com garantias vigentes**", "Média", "Média", "Dano ao erário.", "Quadro de garantias por lote (Seção 11, item 3). *(Resp.: Fiscal)*", "Glosa dos valores. *(Resp.: Fiscal/Gestor)*"],
    ["**9. Indisponibilidade do local da visita técnico-pedagógica**", "Média", "Baixa", "Frustração da etapa de culminância do programa.", "Cláusula de local alternativo equivalente. *(Resp.: Gestor)*", "Substituição aprovada pela fiscalização. *(Resp.: Fiscal)*"],
    ["**10. Dependência tecnológica e perda de dados ao término do contrato**", "Média", "Alta", "Perda de histórico e de indicadores.", "Cláusula de portabilidade em formato aberto e plano de transição. *(Resp.: Gestor)*", "Execução do plano de transição contratual. *(Resp.: Gestor)*"],
    ["**11. Tratamento de dados de crianças e adolescentes**", "Baixa", "Alta", "Violação à LGPD e dano institucional.", "Base legal (LGPD, art. 14), termo de autorização e acordo de tratamento de dados com a contratada. *(Resp.: Gestor)*", "Suspensão do tratamento e comunicação à ANPD. *(Resp.: Gestor/Encarregado)*"],
]
rep(anchor10, anchor10 + "\n" + "\n".join("| " + " | ".join(I(c) for c in r) + " |" for r in rows), "M-14a")
rep("*(Nota: O Risco 6 referente à descontinuidade orçamentária anual está referenciado nas Seções 1 e 13 conforme art. 106, II, da Lei nº 14.133/2021).*",
    D("*(Nota: O Risco 6 referente à descontinuidade orçamentária anual está referenciado nas Seções 1 e 13 conforme art. 106, II, da Lei nº 14.133/2021).*") + " " + I("*(Nota: O Risco 6 trata da descontinuidade orçamentária, nos termos do art. 106, II e III, da Lei nº 14.133/2021, também referida na Seção 1.)*"), "M-14b")

# ---------- Seção 11 (A-04)
rep("MDM nem formação pedagógica.",
    "MDM nem formação pedagógica.\n3. " + I("**Garantias contratuais vigentes dos Chromebooks adquiridos entre 2022 e 2025:** " + ph("preencher por lote: processo, quantidade, data de recebimento, prazo de garantia, término") + ". Os equipamentos em garantia permanecem sob responsabilidade do respectivo fornecedor até o término do prazo, e a contratada somente passará a atendê-los após esse término, mediante inclusão formal na base de cobertura, de modo a afastar pagamento em duplicidade. Os 7 (sete) equipamentos do lote 2025 registrados como inoperantes (Anexo II, item 6) devem ser encaminhados de imediato à garantia do fornecedor."), "A-04a")
rep("**Conclusão:** Não há sobreposição de objetos ou duplicidade de pagamento; a contratação é complementar e tecnicamente distinta.",
    "**Conclusão:** Não há sobreposição de objetos ou duplicidade de pagamento" + I(", observado o tratamento das garantias vigentes previsto no item 3") + "; a contratação é complementar e tecnicamente distinta.", "A-04b")

# ---------- Seção 13 (M-21, A-03, M-16, A-08, C-01)
rep("sujeito à validação jurídica prévia.",
    "sujeito à validação jurídica prévia." + I(" Os serviços enquadram-se como comuns (art. 6º, XIII), pois seus padrões de desempenho e qualidade, inclusive os do Programa Aluno Tutor (carga horária, eixos, produtos e certificação), são objetivamente definidos no Termo de Referência por especificações usuais de mercado."), "M-21")
rep("* **Formato e Vigência:** **Lote único**, pelo prazo plurianual de **48 (quarenta e oito) meses** (art. 106 da Lei nº 14.133/2021), com previsão de reajuste anual por índice setorial (ex.: IPCA) no Termo de Referência (art. 92).",
    "* **Formato e Vigência:** " + D("**Lote único**") + " " + I("**Grupo 1 (itens 1, 2 e 3) e itens 4 e 5 com adjudicação autônoma** (Seção 5.2)") + ", pelo prazo plurianual de **48 (quarenta e oito) meses** (art. 106 da Lei nº 14.133/2021), com previsão de reajuste anual " + D("por índice setorial (ex.: IPCA) no Termo de Referência (art. 92).") + " " + I("por índice setorial de tecnologia da informação (ICTI/IPEA), limitado à variação do IPCA/IBGE no mesmo período, com data-base vinculada à data do orçamento estimado (art. 25, § 7º, e art. 92 da Lei nº 14.133/2021).") + "\n" +
    "* " + I("**Vantajosidade da vigência plurianual (art. 106, I):** a vigência de 48 meses é a mais vantajosa porque (i) dilui os custos de mobilização, implantação do Service Desk, inventário inicial e integração das plataformas; (ii) permite o planejamento de estoque de peças, com reflexo esperado no preço; (iii) coincide com 4 ciclos anuais completos do Programa Aluno Tutor; e (iv) " + ph("preencher: comparação de preço anual × plurianual nas cotações, se obtida") + ". A autoridade competente atestará essa vantagem antes da autorização da contratação."), "A-03d/M-16/A-08")
rep("* **Declaração de Viabilidade:** Conclui-se pela **VIABILIDADE TÉCNICA E ECONÔMICA** da contratação pretendida.",
    "* **Declaração de Viabilidade:** " + D("Conclui-se pela **VIABILIDADE TÉCNICA E ECONÔMICA** da contratação pretendida.") + " " + I("Conclui-se pela **VIABILIDADE TÉCNICA** da contratação. A **viabilidade econômica** é demonstrada, em caráter preliminar, pela estimativa da Seção 6, e será confirmada no Termo de Referência após a conclusão da pesquisa de preços formal (Anexo III), ficando o prosseguimento condicionado a essa confirmação."), "C-01c")

# ---------- Anexo I (A-26 Opção 1)
rep("Nos termos do art. 8º do Decreto Municipal nº 4.345/2023, o presente Estudo Técnico Preliminar foi elaborado por:",
    "Nos termos do art. 8º" + I(", caput,") + " do Decreto Municipal nº 4.345/2023, o presente Estudo Técnico Preliminar foi elaborado " + D("por") + " " + I("conjuntamente pelos servidores das áreas técnica e requisitante") + ":", "A-26a")
rep("* **Cargo:** Diretor do Departamento de Tecnologia Educacional  ",
    "* **Cargo:** Diretor do Departamento de Tecnologia Educacional" + I(" — área técnica") + "  ", "A-26b")
team = """| Nome | Área | Cargo | Matrícula |
| :--- | :--- | :--- | :--- |
| ⟦preencher⟧ | Requisitante — ⟦unidade da SME⟧ | ⟦cargo⟧ | ⟦matrícula⟧ |
| ⟦preencher, se houver⟧ | Apoio — ⟦ex.: administrativo/compras⟧ | ⟦cargo⟧ | ⟦matrícula⟧ |"""
rep("* **Registro:** 5192  \n", "* **Registro:** 5192  \n\n" + wrap_table(team, I) + "\n", "A-26c")

# ---------- Anexo II (B-18)
rep("* **Tela:** 15 ocorrências (32,6%)", "* **Tela:** 15 ocorrências (" + D("32,6%") + " " + I("33,3%") + ")", "B-18a")
rep("* **Teclado:** 14 ocorrências (30,4%)", "* **Teclado:** 14 ocorrências (" + D("30,4%") + " " + I("31,1%") + ")", "B-18b")
rep("* **Sistema:** 8 ocorrências", "* **Sistema:** 8 ocorrências" + I(" (17,8%)"), "B-18c")
rep("* **Bateria:** 6 ocorrências", "* **Bateria:** 6 ocorrências" + I(" (13,3%)"), "B-18d")
rep("* **Outros:** 2 ocorrências", "* **Outros:** 2 ocorrências" + I(" (4,4%)"), "B-18e")

# ---------- Anexo III (A-03, M-20, A-25, M-27)
rep("para os 5 itens em lote único por 48 meses", "para os 5 itens " + D("em lote único") + " " + I("(Grupo 1 e itens 4 e 5)") + " por 48 meses", "A-03e")
rep("mostrou-se plenamente comparável ao escopo completo:",
    "mostrou-se plenamente comparável ao escopo completo" + I(" " + ph("completar a matriz de fontes F1 a F17 — seis fontes não estão nominadas — com órgão, número do processo, data, link do PNCP, objeto, unidade, valor, natureza do valor (estimado, adjudicado, contratado ou registrado em ata), classificação de comparabilidade e motivo da exclusão")) + ":", "M-20")
rep('  * *Item 4 (MDM):* Única referência parcialmente comparável em moeda nacional (TRT 16ª Região, PE 90011/2025 — média de R$ 4,93/licença/mês para 4.140 licenças, sem SLA $8\\times5$, geolocalização ou SecurityScorecard "A").',
    "  * *Item 4 (MDM):* " + D('Única referência parcialmente comparável em moeda nacional (TRT 16ª Região, PE 90011/2025 — média de R$ 4,93/licença/mês para 4.140 licenças, sem SLA $8\\times5$, geolocalização ou SecurityScorecard "A").') + " " +
    I("A contratação do Tribunal Regional do Trabalho da 16ª Região (Pregão Eletrônico nº 90011/2025, UASG 080018, PA SEI nº 000002950/2025, abertura em 04/08/2025) foi examinada e classificada como **inadequada como parâmetro de preço**, pelos seguintes motivos: (i) a solução de MDM integra pacotes de Serviço Móvel Pessoal com smartphone ou tablet em comodato (itens 3 e 6), sem preço segregado para o gerenciamento; (ii) a quantidade de 4.140 refere-se a assinaturas-mês de telefonia (138 linhas × 30 meses), e não a licenças de MDM, que cobrem 144 dispositivos; e (iii) os valores divulgados são unitários máximos estimados (R\\$ 115,70/mês no item 3 e R\\$ 173,30/mês no item 6), e não preços contratados. A referência é mantida apenas como parâmetro técnico: suas funcionalidades mínimas de MDM incluem inventário, restrição de aplicativos, comandos remotos e localização geográfica, e suas práticas contratuais incluem faturamento do MDM em separado, Instrumento de Medição de Resultado e reajuste por índice setorial limitado ao IPCA. Não foi localizada contratação pública com preço de MDM segregado e comparável, razão pela qual a estimativa do item 4 será obtida por cotação direta com no mínimo 3 (três) fornecedores especializados (art. 14, § 4º, do Decreto Municipal nº 4.345/2023), observado, se necessário, o art. 14, § 7º."), "A-25a")
rep("  * *Item 4 (MDM):* Possível multiplicação duplicada da quantidade (R$ 198,00 $\\times$ 40 $\\times$ 48 = R$ 380.160,00 vs. R$ 198,00 $\\times$ 48 = R$ 9.504,00).",
    "  * *Item 4 (MDM):* " + D("Possível multiplicação duplicada da quantidade (R$ 198,00 $\\times$ 40 $\\times$ 48 = R$ 380.160,00 vs. R$ 198,00 $\\times$ 48 = R$ 9.504,00).") + " " + I("É necessário confirmar, por escrito, junto ao fornecedor consultado, se o valor preliminar de R\\$ 198,00 corresponde ao preço por licença/mês (total de R\\$ 198,00 × 40 × 48 = R\\$ 380.160,00) ou ao total mensal das 40 licenças (total de R\\$ 198,00 × 48 = R\\$ 9.504,00)."), "A-25b")
rep("padronizando as propostas por valores mensal, anual e global (48 meses) e exigindo composição detalhada por item.",
    "padronizando as propostas por valores mensal, anual e global (48 meses) e exigindo composição detalhada por item.\n" +
    "  3. " + I("Conceder aos fornecedores consultados prazo de resposta compatível com a complexidade do objeto, de no mínimo " + ph("preencher") + " dias úteis (art. 14, § 4º, I, do Decreto Municipal nº 4.345/2023).") + "\n" +
    "  4. " + I("Registrar nos autos a relação dos fornecedores consultados que não apresentaram proposta, com data e meio de envio da solicitação (art. 14, § 4º, III).") + "\n" +
    "  5. " + I("Calcular o preço estimado de cada item sobre um conjunto de três ou mais preços válidos, adotando-se a mediana quando o coeficiente de variação da amostra superar 25% e a média nos demais casos, com justificativa nos autos (art. 14, caput e § 1º); valores inexequíveis, inconsistentes ou excessivamente elevados serão desconsiderados motivadamente (art. 14, § 6º); na impossibilidade de obter três preços válidos, adotar-se-á a justificativa excepcional do art. 14, § 7º."), "M-27")

# ---------- Anexo IV (C-02)
rep("| Item | Unid. | Descrição dos Itens | Qtde | Valor Unit. Mensal | Valor Total Mensal | Valor Total (48 meses) |",
    "| Item | Unid. | Descrição dos Itens | Qtde | " + D("Valor Unit. Mensal") + " " + I("Preço Unitário (na unidade indicada)") + " | " + D("Valor Total Mensal") + " " + I("Valor Total Anual") + " | Valor Total (48 meses) |", "C-02h")
rep("| **01** | Disp. |", "| **01** | Disp." + I("/mês") + " |", "C-02i")
rep("| **02** | Unid. |", "| **02** | " + D("Unid.") + " " + I("Estação/mês") + " |", "C-02j")
rep("| **03** | Mês |", "| **03** | " + D("Mês") + " " + I("Serviço/mês") + " |", "C-02k")
rep("| **04** | Licença |", "| **04** | Licença" + I("/mês") + " |", "C-02l")
rep("| **05** | Aluno/Turma | Programa Aluno Tutor de Tecnologia (30 alunos/ano) |",
    "| **05** | " + D("Aluno/Turma") + " " + I("Aluno/ano") + " | Programa Aluno Tutor de Tecnologia (30 alunos/ano)" + I(" — preço por aluno em um ciclo anual completo; total = preço × 30 × 4 ciclos") + " |", "C-02m")
rep("* **Data de Entrega do Orçamento:**",
    I("**Instrução ao proponente:** para o item 5, o preço unitário refere-se a um aluno em um ciclo anual completo; para o item 4, a uma licença por mês; para os itens 1 a 3, a um dispositivo, estação ou serviço por mês. Anexar planilha de composição de custos por item (peças, mão de obra, deslocamento, licenciamento e margem).") + "\n\n* **Data de Entrega do Orçamento:**", "C-02n")

# ---------- Anexo V (M-19)
rep("Servidor(es) municipal(is) da frota.", "Servidor(es) municipal(is) da frota" + I(" e educador(es) designado(s) da rede municipal") + ".", "M-19b")
rep("**Bertioga/SP, ____ de ______________ de 2026.**",
    I("Autorizo o tratamento dos dados pessoais do(a) aluno(a) estritamente para a finalidade desta atividade, nos termos do art. 14 da Lei nº 13.709/2018, e ( ) autorizo / ( ) não autorizo o uso de sua imagem em comunicações institucionais da Secretaria Municipal de Educação.") + "\n\n**Bertioga/SP, ____ de ______________ de 2026.**", "M-19c")

# ---------- Conversão dos marcadores para spans do pandoc
def conv(m, style):
    inner = m.group(1).replace("⟦", "\\[").replace("⟧", "\\]")
    return "[" + inner + ']{custom-style="' + style + '"}'
t = re.sub(r"\{\+(.+?)\+\}", lambda m: conv(m, "Inserido"), t, flags=re.S)
t = re.sub(r"\{-(.+?)-\}", lambda m: conv(m, "Excluido"), t, flags=re.S)
assert "⟦" not in t and "{+" not in t and "{-" not in t
# "R$ " fora de math: escapar para evitar interpretação como TeX
t = re.sub(r"R\$(?=\s)", r"R\\$", t)
open(OUT, "w", encoding="utf-8").write(t)
print(len(log), "alterações aplicadas:", ", ".join(log))
