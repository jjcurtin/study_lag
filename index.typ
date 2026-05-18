// Simple numbering for non-book documents
#let equation-numbering = "(1)"
#let callout-numbering = "1"
#let subfloat-numbering(n-super, subfloat-idx) = {
  numbering("1a", n-super, subfloat-idx)
}

// Theorem configuration for theorion
// Simple numbering for non-book documents (no heading inheritance)
#let theorem-inherited-levels = 0

// Theorem numbering format (can be overridden by extensions for appendix support)
// This function returns the numbering pattern to use
#let theorem-numbering(loc) = "1.1"

// Default theorem render function
#let theorem-render(prefix: none, title: "", full-title: auto, body) = {
  if full-title != "" and full-title != auto and full-title != none {
    strong[#full-title.]
    h(0.5em)
  }
  body
}
// Some definitions presupposed by pandoc's typst output.
#let content-to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(content-to-string).join("")
  } else if content.has("body") {
    content-to-string(content.body)
  } else if content == [ ] {
    " "
  }
}

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms.item: it => block(breakable: false)[
  #text(weight: "bold")[#it.term]
  #block(inset: (left: 1.5em, top: -0.4em))[#it.description]
]

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let fields = old_block.fields()
  let _ = fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => {
          let subfloat-idx = quartosubfloatcounter.get().first() + 1
          subfloat-numbering(n-super, subfloat-idx)
        })
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => block({
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          })

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let children = old_title_block.body.body.children
  let old_title = if children.len() == 1 {
    children.at(0)  // no icon: title at index 0
  } else {
    children.at(1)  // with icon: title at index 1
  }

  // TODO use custom separator if available
  // Use the figure's counter display which handles chapter-based numbering
  // (when numbering is a function that includes the heading counter)
  let callout_num = it.counter.display(it.numbering)
  let new_title = if empty(old_title) {
    [#kind #callout_num]
  } else {
    [#kind #callout_num: #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block,
    block_with_new_content(
      old_title_block.body,
      if children.len() == 1 {
        new_title  // no icon: just the title
      } else {
        children.at(0) + new_title  // with icon: preserve icon block + new title
      }))

  align(left, block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1)))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color,
        width: 100%,
        inset: 8pt)[#if icon != none [#text(icon_color, weight: 900)[#icon] ]#title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}


// syntax highlighting functions from skylighting:
/* Function definitions for syntax highlighting generated by skylighting: */
#let EndLine() = raw("\n")
#let Skylighting(fill: none, number: false, start: 1, sourcelines) = {
   let blocks = []
   let lnum = start - 1
   let bgcolor = rgb("#f1f3f5")
   for ln in sourcelines {
     if number {
       lnum = lnum + 1
       blocks = blocks + box(width: if start + sourcelines.len() > 999 { 30pt } else { 24pt }, text(fill: rgb("#aaaaaa"), [ #lnum ]))
     }
     blocks = blocks + ln + EndLine()
   }
   block(fill: bgcolor, width: 100%, inset: 8pt, radius: 2pt, blocks)
}
#let AlertTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let AnnotationTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let AttributeTok(s) = text(fill: rgb("#657422"),raw(s))
#let BaseNTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let BuiltInTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let CharTok(s) = text(fill: rgb("#20794d"),raw(s))
#let CommentTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let CommentVarTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))
#let ConstantTok(s) = text(fill: rgb("#8f5902"),raw(s))
#let ControlFlowTok(s) = text(weight: "bold",fill: rgb("#003b4f"),raw(s))
#let DataTypeTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let DecValTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let DocumentationTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))
#let ErrorTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let ExtensionTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let FloatTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let FunctionTok(s) = text(fill: rgb("#4758ab"),raw(s))
#let ImportTok(s) = text(fill: rgb("#00769e"),raw(s))
#let InformationTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let KeywordTok(s) = text(weight: "bold",fill: rgb("#003b4f"),raw(s))
#let NormalTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let OperatorTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let OtherTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let PreprocessorTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let RegionMarkerTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let SpecialCharTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let SpecialStringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let StringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let VariableTok(s) = text(fill: rgb("#111111"),raw(s))
#let VerbatimStringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let WarningTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))


// document mode
#let doc(
  title: none,
  running-head: none,
  authors: none,
  affiliations: none,
  authornote: none,
  abstract: none,
  keywords: none,
  margin: (x: 2.5cm, y: 2.5cm),
  paper: "us-letter",
  font: ("New Computer Modern"),
  fontsize: 11pt,
  leading: 0.55em,
  spacing: 0.55em,
  first-line-indent: 1.25cm,
  toc: false,
  cols: 1,
  doc,
) = {

  set page(
    paper: paper,
    margin: margin,
    header-ascent: 50%,
    header: locate(
        loc => if [#loc.page()] == [1] {
          []
        } else {
          grid(
            columns: (1fr, 1fr),
            align(left)[#running-head],
            align(right)[#counter(page).display()]
          )
        }
    ),
  )
  
  set par(
    justify: true, 
    leading: leading,
    first-line-indent: first-line-indent
  )

  // Also "leading" space between paragraphs
  show par: set block(spacing: spacing)

  set text(
    font: font,
    size: fontsize
  )

  if title != none {
    align(center)[
      #v(6em)#block(below: leading*4)[
        #text(size: fontsize*1.4)[#title]
      ]
    ]
  }
  
  if authornote != none {
    footnote(numbering: "*", authornote)
    counter(footnote).update(0)
  }
  
  if authors != none {
    align(center)[
      #block(below: leading*2)[
        #set text(size: fontsize*1.15)
        // Formatting depends on N authors 1, 2, or 2+
        #if authors.len() > 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [#if a==authors.at(authors.len()-2) [, and] else [,]]
          ]
        } 
        #if authors.len() == 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [and]
          ]
        }
        #if authors.len() == 1 {
          for a in authors [
            #a.name#super[#a.affiliations]
          ]
        }
      ]
    ]
  }

  if affiliations != none {
    align(center)[
      #block(below: leading*2)[
        #for a in affiliations [
          #super[#a.id]#a.name \
        ]
      ]
    ]
  }

  if abstract != none {
    block(inset: (x: 10%, y: 0%), below: 3em)[
      #align(center, text("Abstract"))
      #set par(first-line-indent: 0pt, leading: leading)
      #abstract
      #if keywords != none {[
        #v(1em)#text(weight: "regular", style: "italic")[Keywords:] #h(0.25em) #keywords
      ]}
    ]
  }

  /* Redefine headings up to level 5 */
  show heading.where(
    level: 1
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(center)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 2
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(left)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 3
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(left)
    #set text(size: fontsize, style: "italic")
    #it.body
  ]

  show heading.where(
    level: 4
  ): it => text(
    size: 1em,
    weight: "bold",
    it.body + [.]
  )

  show heading.where(
    level: 5
  ): it => text(
    size: 1em,
    weight: "bold",
    style: "italic",
    it.body + [.]
  )

  if cols == 1 {
    doc
  } else {
    columns(cols, gutter: 4%, doc)
  }
  
}

// Manuscript mode
#let man(
  title: none,
  running-head: none,
  authors: none,
  affiliations: none,
  authornote: none,
  abstract: none,
  keywords: none,
  margin: (x: 2.5cm, y: 2.5cm),
  paper: "us-letter",
  font: ("Times New Roman"),
  fontsize: 12pt,
  leading: 2em,
  spacing: 2em,
  first-line-indent: 1.25cm,
  toc: false,
  cols: 1,
  doc,
) = {

  set page(
    paper: paper,
    margin: margin,
    header-ascent: 50%,
    header: grid(
      columns: (1fr, 1fr),
      align(left)[#running-head],
      align(right)[#counter(page).display()]
    )
  )
  
  set par(
    justify: false, 
    leading: leading,
    first-line-indent: first-line-indent
  )

  // Also "leading" space between paragraphs
  show par: set block(spacing: spacing)

  set text(
    font: font,
    size: fontsize
  )

  if title != none {
    align(center)[
      #v(8em)#block(below: leading*2)[
        #text(weight: "bold", size: fontsize)[#title]
      ]
    ]
  }
  
  if authornote != none {
    footnote(numbering: "*", authornote)
    counter(footnote).update(0)
  }
  
  if authors != none {
    align(center)[
      #block(above: leading, below: leading)[
        // Formatting depends on N authors 1, 2, or 2+
        #if authors.len() > 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [#if a==authors.at(authors.len()-2) [, and] else [,]]
          ]
        } 
        #if authors.len() == 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [and]
          ]
        }
        #if authors.len() == 1 {
          for a in authors [
            #a.name#super[#a.affiliations]
          ]
        }
      ]
      #counter(footnote).update(0)
    ]
  }
  
  if affiliations != none {
    align(center)[
      #block(above: leading, below: leading)[
        #for a in affiliations [
          #super[#a.id]#a.name \
        ]
      ]
    ]
  }

  pagebreak()
  
  if abstract != none {
    block(above: 0em, below: 2em)[
      #align(center, text(weight: "bold", "Abstract"))
      #set par(first-line-indent: 0pt, leading: leading)
      #abstract
      #if keywords != none {[
        #text(weight: "regular", style: "italic")[Keywords:] #h(0.25em) #keywords
      ]}
    ]
  }
  pagebreak()

  /* Redefine headings up to level 5 */
  show heading.where(
    level: 1
  ): it => block(width: 100%, below: leading, above: leading)[
    #set align(center)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 2
  ): it => block(width: 100%, below: leading, above: leading)[
    #set align(left)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 3
  ): it => block(width: 100%, below: leading, above: leading)[
    #set align(left)
    #set text(size: fontsize, style: "italic")
    #it.body
  ]

  show heading.where(
    level: 4
  ): it => text(
    size: 1em,
    weight: "bold",
    it.body + [.]
  )

  show heading.where(
    level: 5
  ): it => text(
    size: 1em,
    weight: "bold",
    style: "italic",
    it.body + [.]
  )

  if cols == 1 {
    doc
  } else {
    columns(cols, gutter: 4%, doc)
  }
  
}

// Journal mode
#let jou(
  title: none,
  running-head: none,
  authors: none,
  affiliations: none,
  authornote: none,
  abstract: none,
  keywords: none,
  margin: (x: 2.5cm, y: 2.5cm),
  paper: "us-letter",
  font: ("Times New Roman"),
  fontsize: 10pt,
  leading: 0.5em, // Space between lines
  spacing: 0.5em, // Space between paragraphs
  first-line-indent: 0cm,
  toc: false,
  cols: 2,
  mode: none,
  doc,
) = {

  set page(
    paper: paper,
    margin: margin,
    header-ascent: 50%,
    header: locate(
        loc => if [#loc.page()] == [1] {
          []
        } else {
          grid(
            columns: (1fr, 1fr),
            align(left)[#running-head],
            align(right)[#counter(page).display()]
          )
        }
    ),
  )
  
  set par(
    justify: true, 
    leading: leading,
    first-line-indent: first-line-indent
  )

  // Also "leading" space between paragraphs
  show par: set block(spacing: spacing)

  set text(
    font: font,
    size: fontsize
  )

  if title != none {
    align(center)[
      #v(3em)#block(below: leading*4)[
        #text(size: fontsize*1.8)[#title]
      ]
    ]
  }
  
  if authornote != none {
    footnote(numbering: "*", authornote)
    counter(footnote).update(0)
  }
  
  if authors != none {
    align(center)[
      #block(below: leading*2)[
        #set text(size: fontsize*1.3)
        // Formatting depends on N authors 1, 2, or 2+
        #if authors.len() > 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [#if a==authors.at(authors.len()-2) [, and] else [,]]
          ]
        } 
        #if authors.len() == 2 {
          for a in authors [
            #a.name#super[#a.affiliations]#if a!=authors.at(authors.len()-1) [and]
          ]
        }
        #if authors.len() == 1 {
          for a in authors [
            #a.name#super[#a.affiliations]
          ]
        }
      ]
      #counter(footnote).update(0)
    ]
  }
  
  if affiliations != none {
    align(center)[
      #block(below: leading*2)[
        #for a in affiliations [
          #super[#a.id]#a.name \
        ]
      ]
    ]
  }

  if abstract != none {
    block(inset: (x: 15%, y: 0%), below: 3em)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0pt, leading: leading)
      #abstract
      #if keywords != none {[
        #v(1em)#text(weight: "regular", style: "italic")[Keywords:] #h(0.25em) #keywords
      ]}
    ]
  }

  /* Redefine headings up to level 5 */
  show heading.where(
    level: 1
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(center)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 2
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(left)
    #set text(size: fontsize)
    #it.body
  ]

  show heading.where(
    level: 3
  ): it => block(width: 100%, below: leading*2, above: leading*2)[
    #set align(left)
    #set text(size: fontsize, style: "italic")
    #it.body
  ]

  show heading.where(
    level: 4
  ): it => text(
    size: 1em,
    weight: "bold",
    it.body + [.]
  )

  show heading.where(
    level: 5
  ): it => text(
    size: 1em,
    weight: "bold",
    style: "italic",
    it.body + [.]
  )

  if cols == 1 {
    doc
  } else {
    columns(cols, gutter: 4%, doc)
  }
  
}
#let brand-color = (:)
#let brand-color-background = (:)
#let brand-logo = (:)

#set page(
  paper: "us-letter",
  margin: (x: 1.25in, y: 1.25in),
  numbering: "1",
  columns: 1,
)

#show: document => man(
  title: "Forecasting Alcohol Lapse Risk up to Two Weeks in Advance using Time-lagged Machine Learning Models",
  authors: (
    (
      name: "Kendra Wyant",
      affiliations: "aff-1",
      email: []
    ),
    (
      name: "Gaylen E. Fronk",
      affiliations: "aff-1,aff-2",
      email: []
    ),
    (
      name: "Jiachen Yu",
      affiliations: "aff-1",
      email: []
    ),
    (
      name: "Claire E. Punturieri",
      affiliations: "aff-1",
      email: []
    ),
    (
      name: "John J. Curtin",
      affiliations: "aff-1",
      email: [jjcurtin\@wisc.edu]
    ),
    
  ),
  affiliations: (
    (
      id: "aff-1",
      name: "Department of Psychology, University of Wisconsin-Madison"
    ),
    (
      id: "aff-2",
      name: "Department of Psychiatry and Behavioral Sciences, Medical University of South Carolina"
    ),
    
  ),
  abstract: [We developed machine learning models to predict future alcohol lapses within 24-hour windows lagged 1 day, 3 days, 1 week, and 2 weeks. We engineered features from 4x daily ecological momentary assessment from individuals (N=151; 51% male; mean age=41; 87% non-Hispanic White) in early recovery (\<= 8 weeks of abstinence) over three months. We trained and evaluated models using nested cross-validation. Median posterior auROC was high (0.85--0.91) for all models but decreased modestly with increasing lag. Models performed worse for non-advantaged groups (non-White and/or Hispanic, below poverty, female). Past alcohol use, abstinence self-efficacy, and craving were the most important features, with the magnitude of importance varying meaningfully by lag. These findings demonstrate feasibility of predicting next-day lapses up to two weeks in advance. Embedding these models in a recovery monitoring support system could enable adaptive, personalized care. Improving model fairness and optimizing the delivery of model feedback to sustain engagement remain critical next steps.

],
  keywords: [Substance use disordersPrecision mental health],
  document,
)

= Introduction
<introduction>
Alcohol and other substance use disorders (SUDs) are serious chronic conditions, characterized by high relapse rates \[1,2\], substantial co-morbidity with other physical and mental health problems \[2,#strong[substanceabuseandmentalhealthservicesadministration2023NSDUHDetailed?]\], and an increased risk of mortality \[3,4\]. Too few individuals receive medications or clinician-delivered interventions to help them initially achieve abstinence and/or reduce harms associated with their use \[#strong[substanceabuseandmentalhealthservicesadministration2023NSDUHDetailed?]\]. Moreover, this problem is even worse for subsequent continuing care during SUD recovery. Continuing care, including both risk monitoring and ongoing support, is the gold standard for managing chronic health conditions such as diabetes, asthma, and HIV \[5\]. Yet, continuing care for SUDs is largely lacking despite ample evidence that SUDs are chronic, relapsing conditions \[6,7,#strong[substanceabuseandmentalhealthservicesadministration2023NSDUHDetailed?]\].

An important focus of continuing care during SUD recovery is to prevent lapses (i.e., single instances of goal-inconsistent substance use) and full relapse back to harmful use \[8,9\]. Critically, the risk factors that instigate lapses during recovery are individualized, numerous, dynamic, interactive, and non-linear \[10,11\]. The optimal supports to address these risk factors and encourage continued, successful recovery vary both across individuals and within an individual over time. Given this, continuing care could benefit greatly from a precision mental health approach that seeks to provide the right support to the right individual at the right time, every time \[12--14\]. However, such monitoring and personalized support must also be highly scalable to address the substantial unmet need for SUD continuing care.

Recent advances in both smartphone sensing \[15\] and machine learning \[16\] hold promise as a scalable foundation for monitoring and personalized support during SUD recovery. Smartphone sensing approaches (e.g., ecological momentary assessment \[EMA\], geolocation sensing) can provide the frequent, longitudinal measurement of proximal risk factors that is necessary for prediction of future lapses with high temporal precision. EMA may be particularly well-suited for lapse prediction because it can provide privileged access to subjective experiences (e.g., craving, affect, stress, motivation, self-efficacy) that are targets for change in evidence-based approaches for relapse prevention \[8,9,17\]. Furthermore, individuals with SUDs have found EMA to be acceptable for sustained measurement for up to a year with relatively high compliance \[18,19\], suggesting that this method is feasible for long-term monitoring throughout SUD recovery.

Machine learning models are well-positioned to use EMAs as inputs to provide temporally precise prediction of the probability of future lapses with sufficiently high performance to support decisions about interventions and other supports for specific individuals. These models can handle the high dimensional feature sets that may result from feature engineering densely sampled raw EMA over time \[20\]. They can also accommodate non-linear and interactive relationships between features and lapse probability that are likely necessary for accurate prediction of lapse probability. Moreover, rapid advances in the tools for interpretable machine learning (e.g., Shapley values \[21\]) now allow us to probe these models to understand which risk features contribute most strongly to a lapse prediction for a specific individual at a specific moment in time. Interventions, supports, and/or suggested lifestyle adjustments can then be personalized to address these risks following from our understanding about relapse prevention.

Preliminary research is now emerging that uses features derived from EMAs in machine learning models to predict the probability of future alcohol use \[20,22,23\]. This research is important because it rigorously required strict temporal ordering necessary for true prediction, with features measured before alcohol use outcomes. These studies also used resampling methods (e.g., cross-validation) that prioritize model generalizability to increase the likelihood these models will perform well with new people. Perhaps most importantly, \[20\] demonstrated that machine learning models using EMA can provide predictions with very high temporal precision at clinically implementable levels of performance. Specifically, we developed models that predict lapses in the immediate future (i.e., the next day and even the next hour) with areas under the receiver operating characteristic curve (auROCs) of 0.91 and 0.93, respectively.

Wyant et al.'s \[20\] next day lapse prediction model can provide personalized support recommendations to address immediate risks for possible lapses. Features derived from past EMAs can be updated in the early morning to yield the predicted lapse probability for an individual that day. Personalized supports that target the top features contributing to that prediction can then be provided. For example, if predicted lapse probability is high due to frequent craving, the individual could be reminded about the benefits of urge surfing or distracting activities during brief periods when cravings arise. Conversely, guided relaxation techniques could be recommended if lapse probability is high due to recent past and anticipated stressors that day. Patients could also be assisted to implement any of these recommendations using videos or other tools within a digital therapeutic. Curtin and colleagues are currently evaluating outcomes associated with the use of this "smart" (machine learning guided) recovery monitoring and support system (RMSS) for patients with an alcohol use disorder (AUD) \[24\].

Despite the promise offered by a smart RMSS based on immediate future risks (e.g., the next day), such a system has limitations. Most importantly, recommendations must be limited to previously learned skills and/or supports that are available to implement that day. However, many risks may require supports that are not available in the moment. For example, to address lifestyle imbalances, several future positive activities may need to be planned. Time with supportive friends or an AA sponsor may require time to schedule. Similarly, work or family schedules may need to be adjusted to return to attending self-help meetings. If new recovery skills or therapeutic activities are needed to address emerging risks, patients may need to book sessions with a therapist. In all these instances, patients would benefit from advanced warning about changes in their lapse probability and the associated risks that contribute to these changes. A smart RMSS could provide this advanced warning by lagging lapse probability predictions further into the future (e.g., predicting lapse probability in a 24-hour window that begins two weeks in the future). However, we do not know if such lagged models could maintain adequate performance for clinical implementation.

In this study, we evaluated the performance of machine learning models that predict the probability of future lapses within 24-hour prediction windows that were systematically lagged further into the future. We considered several meaningful lags for these prediction windows: 1 day, 3 days, 1 week, and 2 weeks. We conducted pre-registered analyses of both the absolute performance of these lagged models and their relative performance compared to a baseline model that predicted lapse probability in the immediate next day (i.e., no lag). In addition to the aggregate performance of these models, we also evaluated algorithmic fairness by comparing model performance across important subgroups that have documented disparities in treatment access and/or outcomes. These include comparisons by race/ethnicity \[25,26\], income \[27\] and sex at birth \[26,28\]. Finally, we calculated Shapley values for feature categories defined by EMA items to better understand how these models generate predictions and how these features can be used to tailor personalized supports.

= Methods
<methods>
== Transparency
<transparency>
We adhere to research transparency principles that are crucial for robust and replicable science. We preregistered our data analytic strategy. We reported all data exclusions. Our data, questionnaires, preregistration, and other study materials are publicly available on our OSF page (#link("https://osf.io/xta67/")). Our annotated analysis scripts and results are publicly available on our study website (#link("https://jjcurtin.github.io/study_lag/")).

== Participants
<participants>
We recruited 192 participants in early recovery (\<= 8 weeks of abstinence) from AUD in Madison, Wisconsin, USA for a 3-month longitudinal study from February 15, 2017 through September 19, 2019. This sample size was determined based on traditional power analysis methods for logistic regression \[29\] because comparable approaches for machine learning models have not yet been validated. Participants were recruited through print and targeted digital advertisements and partnerships with treatment centers. We required that participants:

+ were age 18 or older,
+ could write and read in English,
+ had at least moderate AUD (\>= 4 self-reported DSM-5 symptoms),
+ were abstinent from alcohol for 1-8 weeks, and
+ were willing to use a single smartphone (personal or study provided) while on study.

We also excluded participants exhibiting severe symptoms of psychosis or paranoia (Defined as scores \>2.2 or 2.8, respectively, on the psychosis or paranoia scales of the Symptom Checklist--90 \[30\])

Of the 192 eligible participants, 191 consented to participate in the study at the screening visit, and 169 subsequently enrolled in the study at the enrollment visit, which occurred approximately one week later. Fifteen participants discontinued before the first monthly follow-up visit. We excluded data from one participant who did not maintain a goal of abstinence during their participation. We also excluded data from two participants due to evidence of careless responding and unusually low compliance. Our final sample consisted of 151 participants.

== Procedure
<procedure>
Participants completed five study visits over approximately three months. After an initial phone screen, participants attended an in-person screening visit to determine eligibility, complete informed consent, and collect self-report measures. Eligible, consented participants returned approximately one week later for an intake visit. Three additional follow-up visits occurred about every 30 days that participants remained on study. Participants were expected to complete four daily EMAs. Other personal sensing data streams (geolocation, cellular communications, sleep quality, and audio check-ins) were collected as part of the parent grant's aims (R01 AA024391).

Participants were compensated \$20 per hour for all time spent in the laboratory (i.e., during screening, intake, and follow-up visits). In addition, participants received a \$99 bonus upon completing the full 3-month study. They were also provided \$66 per month to offset costs associated with their cellular plans. Participants received \$25 for each month in which they provided EMA data with less than 10% missingness. Additional compensation was provided for other personal sensing data streams if minimum compliance thresholds were met.

== Ethics
<ethics>
All procedures were approved by the University of Wisconsin-Madison Institutional Review Board (Study \#2015-0780) and carried out in accordance with the principles of the Declaration of Helsinki. All participants provided written informed consent observed by a research assistant.

== Measures
<measures>
=== Ecological Momentary Assessments
<ecological-momentary-assessments>
Participants completed four brief (7-10 questions) EMAs daily. The first and last EMAs of the day were scheduled within one hour of participants' typical wake and sleep times. The other two EMAs were scheduled randomly within the first and second halves of their typical day, with at least one hour between EMAs. Participants learned how to complete the EMA and reviewed the meaning of each question with a member of the research team during their intake visit to ensure consistent question interpretation.

On the first item of all EMAs, participants reported the start and dates and times of any unreported past alcohol use. Next, participants rated the maximum intensity of recent (i.e., since last EMA) experiences of craving, risky situations, stressful events, and pleasant events. Finally, participants rated their current affect on two bipolar scales: valence (Unpleasant/Unhappy to Pleasant/Happy) and arousal (Calm/Sleepy to Aroused/Alert). On the first EMA each day, participants also rated anticipated risky situations, stressful events, and the likelihood that they would drink alcohol in the next week (i.e., abstinence self-efficacy).

=== Individual Characteristics
<individual-characteristics>
We collected self-report information about demographics (age, sex at birth, race, ethnicity, education, marital status, employment, and income) and clinical characteristics (AUD milestones, number of quit attempts, lifetime AUD treatment history, lifetime receipt of AUD medication, DSM-5 AUD symptom count, current drug use \[31\], and presence of psychological symptoms \[30\] to characterize our sample. DSM-5 AUD symptom count and presence of psychological symptoms were also used to determine eligibility. Demographics were included as features in our models. A subset of these variables (sex at birth, race, ethnicity, and income) were used for model fairness analyses, as they have documented disparities in treatment access and outcomes. As part of the aims of the parent project, we collected many other trait and state measures throughout the study. A complete list of all measures can be found on our study's OSF page (#link("https://osf.io/xta67/")).

== Data Analytic Strategy
<data-analytic-strategy>
Data preprocessing, modeling, and Bayesian analyses were done in R (version 4.4.2) using the tidymodels ecosystem \[32--34\]. Models were trained and evaluated using high-throughput computing resources provided by the University of Wisconsin Center for High Throughput Computing \[35\].

=== Predictions
<predictions>
A #emph[prediction timepoint] (#ref(<fig-method>, supplement: [Figure]), Panel A) is the hour at which our model calculates a predicted probability of a lapse within a future 24-hour prediction window for any specific individual. We calculated features from all available EMAs up until, but not including, the prediction timepoint. The first prediction timepoint for each participant was 24 hours from midnight on their study start date. This ensured at least 24 hours of past EMAs were available. Subsequent prediction timepoints for each participant repeatedly rolled forward hour-by-hour until the end of their study participation.

The #emph[prediction window] (#ref(<fig-method>, supplement: [Figure]), Panel B) spans a period of time in which a lapse might occur. The prediction window width for all models was 24 hours (i.e., models predicted the probability of a lapse occurring within a specific 24-hour period). Prediction windows rolled forward hour-by-hour with the prediction timepoint. However, there were five possible #emph[lag times] between the prediction timepoint and start of the associated prediction window. A prediction window either started immediately after the prediction time point (no lag) or was lagged by 1 day, 3 days, 1 week, or 2 weeks.

Given this structure, our models provided hour-by-hour predicted probabilities of an alcohol lapse in a future 24-hour period. Depending on the model, that future period (the prediction window) might start immediately after the prediction timepoint or up to 2 weeks into the future. For example, at midnight on the 30th day of participation, features would be calculated from the past 30 days of EMAs. Separate models would predict the probability of lapse for 24-hour periods starting at midnight that day, or 24-hour periods starting 1 day, 3 days, 1 week or 2 weeks after midnight on day 30.

#block[
#block[
#figure([
#Skylighting(([#NormalTok("Unable to display output for mime type(s): image/tiff");],));
], caption: figure.caption(
position: bottom, 
[
Panel A shows the prediction timepoints at which our model calculated a predicted probability of a lapse. All available data up until, but not including, the prediction timepoint was used to generate these predictions. Features were created for varying feature scoring epochs before the prediction timepoint (i.e., 12, 24, 48, 72, and 168 hours). Prediction timepoints were updated hourly. Panel B shows how the prediction window (i.e., window in which a lapse might occur) rolls forward hour-by-hour with the prediction timepoint. The prediction window width for all models was 24 hours. Additionally, there were five possible lag times between the prediction timepoint and start of the prediction window. A prediction window either started immediately after the prediction timepoint (no lag) or was lagged by 1 day, 3 days, 1 week, or 2 weeks.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-method>


]
]
=== Labels
<labels>
The start and end dates and times of past drinking episodes were reported on the first EMA item. A prediction window was labeled #emph[lapse] if the start date/hour of any drinking episode fell within that window. A window was labeled #emph[no lapse] if no alcohol use occurred within that window +/- 24 hours. If no alcohol use occurred within the window but did occur within 24 hours of the start or end of the window, the window was excluded. We used this conservative 24-hour fence for labeling windows as no lapse (vs.~excluded) to increase the fidelity of these labels. Given that most windows were labeled no lapse, and the outcome was highly unbalanced, it was not problematic to exclude some no lapse events to further increase confidence in those labels.

This method produced totals of: 274,179 labels for the baseline (no lag) model; 270,911 labels for the 1-day lagged model; 264,362 labels for the 3-day lagged model; 251,458 labels for the 1-week lagged model; and 228,420 labels for the 2-week lagged model.

=== Feature Engineering
<feature-engineering>
Features were calculated using only data collected prior to each prediction timepoint to ensure our models were making true future predictions. For the no lag model, the prediction timepoint was at the start of prediction window, so all data prior to the start of the prediction window were included. For the lagged models, the prediction timepoint was 1 day, 3 days, 1 week, or 2 weeks prior to the start of the prediction window, so the last EMA data used for feature engineering were collected 1 day, 3 days, 1 week, or 2 weeks prior to the start of the prediction window.

A total of 285 features were derived from three data sources:

+ #emph[Prediction window]: We dummy-coded features for day of the week at the start of the prediction window.

+ #emph[Demographics]: We created quantitative features for age (in years) and personal income (in dollars), and dummy-coded features for sex at birth (male vs.~female), race/ethnicity (non-Hispanic White vs.~non-White and/or Hispanic), marital status (married vs.~not married vs.~other), education (high school or less vs.~some college vs.~college degree), and employment status (employed vs.~unemployed).

+ #emph[Previous EMA responses]: We calculated raw and difference features in varying feature scoring epochs (i.e., 12, 24, 48, 72, and 168 hours) before the prediction timepoint for all EMA items, except the alcohol use question. Raw features included min, max, median, and most recent scores for each EMA item across all EMAs in each epoch for a given participant. We calculated difference features by subtracting the participant's mean value (using all available data prior to the prediction window) from the associated raw feature value to capture participant-level changes from baseline.

We calculated raw and difference rate features for past use from reported lapses on the EMA. Raw past use features were generated by dividing the total number of previously reported lapses within a feature scoring epoch by the duration of that epoch. For difference past use rate features, we subtracted the baseline past use rate for that participant (i.e., total number of lapses while on study divided by total hours on study before the prediction window) from their associated raw past use rate. We also calculated raw and difference rate features for completed EMAs (i.e., number of full EMA surveys that were completed in a feature scoring epoch divided by the duration of that epoch).

Features had missing values if the participant did not respond to the relevant EMA question during the associated scoring epoch. The proportion of missing values across features and models was low (median = .02, range = 0 - .13). We imputed missing data using median imputation for numeric features and mode imputation for nominal features. We selected coarse median/mode methods for handling missing data due to the computational costs associated with more advanced forms of imputation (e.g., KNN imputation, multiple imputation). Importantly, our imputation calculations are done using only held-in data and can be applied to any new observation.

Other generic feature engineering steps included removing zero and near-zero variance features as determined from held-in data. A sample feature engineering script (i.e., tidymodels recipe) containing all feature engineering steps is available on our OSF study page.

=== Model Configurations
<model-configurations>
We trained and evaluated five separate classification models: one baseline (no lag) model and one model for 1-day, 3-day, 1-week, and 2-week lagged predictions. We considered four well-established statistical algorithms (elastic net, XGBoost, regularized discriminant analysis, and single layer neural networks) that vary across characteristics expected to affect model performance (e.g., flexibility, complexity, handling higher-order interactions natively) \[36\]. Candidate model configurations differed across sensible values for key hyperparameters. Configurations also differed on outcome resampling method (i.e., no resampling and up-sampling and down-sampling of the outcome using majority/no lapse to minority/lapse ratios ranging from 5:1 to 1:1).

=== Cross-validation
<cross-validation>
We used participant-grouped, nested cross-validation for model training, selection, and evaluation with auROC. auROC indexes the probability that the model will predict a higher score for a randomly selected positive case (lapse) relative to a randomly selected negative case (no lapse). Grouped cross-validation assigns all data from a participant as either held-in or held-out to avoid bias introduced when predicting a participant's data from their own data. Folds were stratified on a between-subject variable of low vs.~high lapsers (low lapsers reported fewer than 10 lapses while on study, and high lapsers reported 10 or more lapses while on study). We used 2 repeats of 5-fold cross-validation for the inner loops (i.e., #emph[validation] sets) and 6 repeats of 5-fold cross-validation for the outer loop (i.e., #emph[test] sets). Best model configurations were selected using median auROC across the 10 validation sets. Final performance evaluation of those best model configurations used median auROC across the 30 test sets.

=== Bayesian Model
<bayesian-model>
We used a Bayesian hierarchical generalized linear model to estimate the posterior probability distributions and 95% Bayesian credible intervals (CIs) from the 30 held-out test sets for our five best models. Following recommendations from the rstanarm team and others \[37,38\], we used the rstanarm default autoscaled, weakly informative, data-dependent priors that take into account the order of magnitude of the variables to provide some regularization to stabilize computation and avoid over-fitting. Priors were set as follows: Residual SD \~ exponential(2.6); intercept (centered predictors) \~ normal(2, 0.98); fixed effects \~ normal(0, 2.43); covariance \~ decov(1, 1, 1, 1). We set two random intercepts to account for our resampling method: one for the repeat, and another for the fold nested within repeat. We specified two sets of pre-registered contrasts for model comparisons. The first set compared each lagged model to the baseline no lag model (1-day lag vs.~no lag, 3-day lag vs.~no lag, 1-week lag vs.~no lag, 2-week lag vs.~no lag). The second set compared adjacently lagged models (3-day lag vs.~1-day lag, 1-week lag vs.~3-day lag, 2-week lag vs.~1-week lag). auROCs were transformed using the logit function and regressed as a function of model contrast.

From the Bayesian model we obtained the posterior distribution (transformed back from logit) and Bayesian CIs for auROCs for all five models. To evaluate our models' overall performance, we report the median posterior probability for auROC and Bayesian CIs. This represents our best estimate for the magnitude of the auROC parameter for each model. If the CIs do not contain .5 (chance performance), this provides strong evidence (\> .95 probability) that our model is capturing signal in the data.

We then conducted Bayesian model comparisons using our two sets of contrasts - baseline and adjacent lags. For both model comparisons, we determined the probability that the models' performances differed systematically from each other. We also report the precise posterior probability for the difference in auROCs and the 95% Bayesian CIs.

=== Fairness Analyses
<fairness-analyses>
Using the same 30 held-out test sets, we calculated the median posterior probability and 95% Bayesian CI for auROC for each model separately by race/ethnicity (non-White and/or Hispanic vs.~non-Hispanic White), income (below poverty line vs.~above poverty line, and sex at birth (female vs.~male). We used course race/ethnicity groupings due to the limited diversity of these demographics in our sample. The poverty cutoff was defined from the 2024 federal poverty line for the 48 contiguous United States. Participants at or below \$15,060 annual income were categorized as below poverty. We conducted Bayesian group comparisons to assess the likelihood that each model performs differently by group. We summarize the differences in posterior probabilities for auROC across models. Individual Bayesian fairness contrasts for all five models are available in the supplement.

=== Model Calibration
<model-calibration>
To further characterize and understand our models, we used our inner resampling procedure (2 repeats of 5-fold cross validation grouped on participant and stratified by high/low lapsers) on the full data set to select a single best model configuration for each classification model (no lag, 1-day, 3-day, 1-week, and 2-week lag). The final configuration selected for each model represents the most reliable and robust configuration for deployment.

The best model configuration for each classification model was fit on the full data set. We fit this configuration using single 5-fold cross-validation. This method allowed us to obtain a single predicted probability for each observation, while still using separate data for model training and prediction. We calibrated our probabilities using Platt scaling \[39\]. We calculated Brier scores to assess the accuracy of our raw and calibrated probabilities for the no lag and 2-week lagged models. Brier scores range from 0 (perfect accuracy) to 1 (perfect inaccuracy). A table of Brier scores for all five models is available in the supplement. We provide calibration plots for the no lag and 2-week lagged models (calibration plots for all five models are available in the supplement).

=== Global Feature Importance
<global-feature-importance>
We used the same single 5-fold cross-validation procedure to calculate raw Shapley values for observations in our held-out folds. Raw Shapley values index the importance of any feature (or set/category of features as described below) to any single prediction for a specific observation (i.e., for a specific 24-hour window for a specific participant), which indicates the "local importance" of that feature \[21\]. More precisely, the magnitude of the raw Shapley value for any feature indicates how much the feature score for that observation adjusted the prediction (in log-odds units) for that observation relative to the mean prediction across all observations. Positive Shapley values indicate that the feature score increased the prediction for that observation and negative values indicate that the feature score decreased the prediction.

Raw Shapley values are additive across features for an observation, with their sum across features equal to the total adjustment of the predicted value for that observation vs.~the mean predicted value across all observations. This property allows raw Shapley values to be added together across features within a category to index the importance of that feature category. We created feature categories by summing raw Shapley values for all features associated with specific EMA items. In three instances, we combined features across two similar EMA items (i.e., past and anticipated risky situations, past and anticipated stressful events, and affective valence and arousal) to yield seven feature categories for distinct constructs assessed by the original 10 EMA items. Specifically, we calculated Shapley values for past use, craving, affective state, past/anticipated risky situations, past/anticipated stressful events, past pleasant events, and abstinence self-efficacy.

We summarized feature importance in two ways. First, we used a traditional approach in which we calculated the global importance of each feature category (mean absolute Shapley value). Global feature importance describes how important a feature is, on average, across all observations from all participants. A large mean absolute Shapley value indicates that a feature category makes substantial contributions to predictions across the dataset. Second, we characterized local feature importance by calculating the proportion of observations in which each feature category had the highest Shapley value. This approach summarizes how frequently a feature category is the most influential contributor to individual predictions. We provided a descriptive plot of the relative ranking of feature categories by their global and local feature importance for the no lag and 2-week lagged models. Feature importance plots for all five models are available in the supplement.

= Results
<results>
== Demographic and Lapse Characteristics
<demographic-and-lapse-characteristics>
#ref(<tbl-demohtml>, supplement: [Table]) provides a detailed breakdown of the demographic and clinical characteristics of our sample (N = 151).

#block[
#figure([
#{set text(font: ("Arial Narrow", "Source Sans Pro", "sans-serif")); table(
  columns: 6,
  align: (left,right,right,left,left,left,),
  table.header(table.cell(align: left)[], table.cell(align: right)[N], table.cell(align: right)[%], table.cell(align: left)[M], table.cell(align: left)[SD], table.cell(align: left)[Range],),
  table.hline(),
  table.cell(align: left)[Age], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[41], table.cell(align: left)[11.9], table.cell(align: left)[21-72],
  table.cell(align: left)[Sex at Birth], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Female], table.cell(align: right)[74], table.cell(align: right)[49.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Male], table.cell(align: right)[77], table.cell(align: right)[51.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Race], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[American Indian/Alaska Native], table.cell(align: right)[3], table.cell(align: right)[2.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Asian], table.cell(align: right)[2], table.cell(align: right)[1.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Black/African American], table.cell(align: right)[8], table.cell(align: right)[5.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[White/Caucasian], table.cell(align: right)[131], table.cell(align: right)[86.8], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Other/Multiracial], table.cell(align: right)[7], table.cell(align: right)[4.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Hispanic, Latino, or Spanish origin], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Yes], table.cell(align: right)[4], table.cell(align: right)[2.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[No], table.cell(align: right)[147], table.cell(align: right)[97.4], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Education], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Less than high school or GED degree], table.cell(align: right)[1], table.cell(align: right)[0.7], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[High school or GED], table.cell(align: right)[14], table.cell(align: right)[9.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Some college], table.cell(align: right)[41], table.cell(align: right)[27.2], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[2-Year degree], table.cell(align: right)[14], table.cell(align: right)[9.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[College degree], table.cell(align: right)[58], table.cell(align: right)[38.4], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Advanced degree], table.cell(align: right)[23], table.cell(align: right)[15.2], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Employment], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Employed full-time], table.cell(align: right)[72], table.cell(align: right)[47.7], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Employed part-time], table.cell(align: right)[26], table.cell(align: right)[17.2], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Full-time student], table.cell(align: right)[7], table.cell(align: right)[4.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Homemaker], table.cell(align: right)[1], table.cell(align: right)[0.7], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Disabled], table.cell(align: right)[7], table.cell(align: right)[4.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Retired], table.cell(align: right)[8], table.cell(align: right)[5.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Unemployed], table.cell(align: right)[18], table.cell(align: right)[11.9], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Temporarily laid off, sick leave, or maternity leave], table.cell(align: right)[3], table.cell(align: right)[2.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Other, not otherwise specified], table.cell(align: right)[9], table.cell(align: right)[6.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Personal Income], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[\$34,298], table.cell(align: left)[\$31,807], table.cell(align: left)[\$0-200,000],
  table.cell(align: left)[Marital Status], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Never married], table.cell(align: right)[67], table.cell(align: right)[44.4], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Married], table.cell(align: right)[32], table.cell(align: right)[21.2], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Divorced], table.cell(align: right)[45], table.cell(align: right)[29.8], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Separated], table.cell(align: right)[5], table.cell(align: right)[3.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Widowed], table.cell(align: right)[2], table.cell(align: right)[1.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[DSM-5 Alcohol Use Disorder Symptom Count], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[8.9], table.cell(align: left)[1.9], table.cell(align: left)[4-11],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[Alcohol Use Disorder Milestones],
  table.cell(align: left)[Age of first drink], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[14.6], table.cell(align: left)[2.9], table.cell(align: left)[6-24],
  table.cell(align: left)[Age of regular drinking], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[19.5], table.cell(align: left)[6.6], table.cell(align: left)[11-56],
  table.cell(align: left)[Age at which drinking became problematic], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[27.8], table.cell(align: left)[9.6], table.cell(align: left)[15-60],
  table.cell(align: left)[Age of first quit attempt], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[31.5], table.cell(align: left)[10.4], table.cell(align: left)[15-65],
  table.cell(align: left)[Number of Quit Attempts\*], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[5.5], table.cell(align: left)[5.8], table.cell(align: left)[0-30],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[Lifetime History of Treatment (Can choose more than 1)],
  table.cell(align: left)[Long-term residential (6+ months)], table.cell(align: right)[8], table.cell(align: right)[5.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Short-term residential (\< 6 months)], table.cell(align: right)[49], table.cell(align: right)[32.5], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Outpatient], table.cell(align: right)[74], table.cell(align: right)[49.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Individual counseling], table.cell(align: right)[97], table.cell(align: right)[64.2], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Group counseling], table.cell(align: right)[62], table.cell(align: right)[41.1], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Alcoholics Anonymous/Narcotics Anonymous], table.cell(align: right)[93], table.cell(align: right)[61.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Other], table.cell(align: right)[40], table.cell(align: right)[26.5], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[Received Medication for Alcohol Use Disorder],
  table.cell(align: left)[Yes], table.cell(align: right)[59], table.cell(align: right)[39.1], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[No], table.cell(align: right)[92], table.cell(align: right)[60.9], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[Current (Past 3 Month) Drug Use],
  table.cell(align: left)[Tobacco products (cigarettes, chewing tobacco, cigars, etc.)], table.cell(align: right)[84], table.cell(align: right)[55.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Cannabis (marijuana, pot, grass, hash, etc.)], table.cell(align: right)[66], table.cell(align: right)[43.7], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Cocaine (coke, crack, etc.)], table.cell(align: right)[18], table.cell(align: right)[11.9], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Amphetamine type stimulants (speed, diet pills, ecstasy, etc.)], table.cell(align: right)[15], table.cell(align: right)[9.9], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Inhalants (nitrous, glue, petrol, paint thinner, etc.)], table.cell(align: right)[3], table.cell(align: right)[2.0], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Sedatives or sleeping pills (Valium, Serepax, Rohypnol, etc.)], table.cell(align: right)[22], table.cell(align: right)[14.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Hallucinogens (LSD, acid, mushrooms, PCP, Special K, etc.)], table.cell(align: right)[14], table.cell(align: right)[9.3], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Opioids (heroin, morphine, methadone, codeine, etc.)], table.cell(align: right)[16], table.cell(align: right)[10.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Reported 1 or More Lapse During Study Period], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 6, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[Yes], table.cell(align: right)[84], table.cell(align: right)[55.6], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[No], table.cell(align: right)[67], table.cell(align: right)[44.4], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(align: left)[Number of reported lapses], table.cell(align: right)[], table.cell(align: right)[], table.cell(align: left)[6.8], table.cell(align: left)[12], table.cell(align: left)[0-75],
  table.hline(),
  table.footer([#text(style: "italic")[Note: ]], [], [], [], [], [],
    [#super[] N = 151], [], [], [], [], [],
    [#super[] \*Two participants reported 100 or more quit attempts. We removed these outliers prior to calculating the mean (M), standard deviation (SD), and range.], [], [], [], [], [],),
)}
], caption: figure.caption(
position: top, 
[
Demographic and Clinical Characteristics
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-demohtml>


]
== Model Evaluation
<model-evaluation>
#ref(<fig-pp>, supplement: [Figure]) presents the full posterior probability distributions for auROC for each model (no lag, 1-day, 3-day, 1-week, and 2-week lag). The median auROCs from these posterior distributions were 0.91 (no lag), 0.89 (1-day lag), 0.88 (3-day lag), 0.87 (1-week lag), and 0.85 (2-week lag). These values represent our best estimates for the magnitude of the auROC parameter for each model. The 95% Bayesian CI for the auROCs for these models were relatively narrow and did not contain 0.5: no lag \[0.90-0.92\], 1-day lag \[0.88-0.90\], 3-day lag \[0.87-0.90\], 1-week lag \[0.85-0.88\], 2-week lag \[0.83-0.87\].

#block[
#figure([
#box(image("index_files/figure-typst/notebooks-mak_figures-fig-pp-output-1.png"))
], caption: figure.caption(
position: bottom, 
[
Posterior probability distributions for area under ROC curve (auROC) for each model (no lag, 1-day, 3-day, 1-week, and 2-week lag). Each distribution reflects 12,000 posterior samples (4 chains × 3,000 samples) from a Bayesian hierarchical generalized linear model. Horizontal lines depict 95% Bayesian credible intervals (CI) and vertical solid lines depict median posterior probability for auROC. Vertical dashed line represents expected performance from a random classifier (.5 auROC).
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-pp>


]
== Model Comparisons
<model-comparisons>
#ref(<tbl-model>, supplement: [Table]) presents the median difference in auROC, 95% Bayesian CI, and posterior probability that that the auROC difference was smaller than 0 for all baseline and adjacent lag contrasts. Median auROC differences less than 0 indicate the more lagged model, on average, performed worse than the more immediate model (e.g., 1-day lag -- no lag, 3-day lag -- 1-day lag). There was strong evidence (probabilities = 1) that the lagged models performed worse than the baseline (no lag) model, with average drops in auROC ranging from 0.02-0.06, and the previous adjacent lagged model, with average drops in auROC ranging from 0.01-0.02.

#block[
#figure([
#{set text(font: ("Arial Narrow", "Source Sans Pro", "sans-serif")); table(
  columns: 4,
  align: (left,left,left,left,),
  table.header(table.cell(align: left)[Contrast], table.cell(align: left)[Median], table.cell(align: left)[Bayesian CI], table.cell(align: left)[Probability],),
  table.hline(),
  table.cell(align: left)[Baseline Contrasts], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 4, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[1-day vs. No lag], table.cell(align: left)[-0.021], table.cell(align: left)[\[-0.025, -0.017\]], table.cell(align: left)[1],
  table.cell(align: left)[3-day vs. No lag], table.cell(align: left)[-0.03], table.cell(align: left)[\[-0.035, -0.025\]], table.cell(align: left)[1],
  table.cell(align: left)[1-week vs. No lag], table.cell(align: left)[-0.042], table.cell(align: left)[\[-0.048, -0.037\]], table.cell(align: left)[1],
  table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[2-week vs. No lag], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[-0.062], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[\[-0.07, -0.056\]], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[1],
  table.cell(align: left)[Adjacent Contrasts], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 4, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[3-day vs. 1-day], table.cell(align: left)[-0.009], table.cell(align: left)[\[-0.013, -0.005\]], table.cell(align: left)[1],
  table.cell(align: left)[1-week vs. 3-day], table.cell(align: left)[-0.012], table.cell(align: left)[\[-0.017, -0.008\]], table.cell(align: left)[1],
  table.cell(align: left)[2-week vs. 1-week], table.cell(align: left)[-0.02], table.cell(align: left)[\[-0.026, -0.015\]], table.cell(align: left)[1],
  table.hline(),
  table.footer([#super[] Median auROC differences less than 0 indicate the more lagged model, on average, performed worse than the more immediate model (e.g., 1-day lag - no lag, 3-day lag - 1-day lag). Bayesian CI represents the range of values where there is a 95% probability that the true auROC difference lies within that range. Probability indicates the posterior probability that this difference is smaller than 0 (i.e., the models are performing differently).], [], [], [],),
)}
], caption: figure.caption(
position: top, 
[
Median difference in auROC, 95% Bayesian credible interval (CI), and posterior probability that that the auROC difference was smaller than 0 for all baseline and adjacent lag contrasts.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-model>


]
== Fairness Analyses
<fairness-analyses-1>
#ref(<tbl-fairness>, supplement: [Table]) presents the median difference in auROC, 95% Bayesian CI, and posterior probability that the auROC difference was smaller than 0 for the three fairness contrasts: race/ethnicity (non-White and/or Hispanic; #emph[N] = 20 vs.~Non-Hispanic White; #emph[N] = 131), sex at birth (female; #emph[N] = 74 vs.~male; #emph[N] = 77), and income (below poverty line; #emph[N] = 49 vs.~above poverty line; #emph[N] = 102). Median auROC differences less than 0 indicate the model, on average, performed worse for the non-advantaged group (female, non-White and/or Hispanic, below poverty line) compared to the advantaged group (male, non-Hispanic White, and above poverty line). In #ref(<tbl-fairness>, supplement: [Table]) we present fairness analyses for our baseline model (no lag) and for our longest lagged model (2-week lag), as this is likely the most clinically useful lagged model for providing advanced warning of lapse risk. Fairness analyses for all five models are available in the supplement.

There was strong evidence (probabilities \> .84) that our models performed worse for the non-advantaged groups compared to the advantaged groups. On average, across all five models, there was a median decrease in auROC of 0.13 (range 0.13-0.17) for participants who were non-White and/or Hispanic compared to participants who were non-Hispanic White. On average, across all five models, there was a median decrease in auROC of 0.05 (range 0.04-0.10) for female participants compared to male participants. On average, across all five models, there was a median decrease in auROC of 0.02 (range 0.01-0.04) for participants below the federal poverty line compared to participants above the federal poverty line.

The proportion of positive lapse labels over all labels (lapse and no lapse) for each demographic subgroup were relatively consistent across groups: race/ethnicity (6%, non-White and/or Hispanic vs.~8%, non-Hispanic White), income (12%, below poverty line vs.~7%, above poverty line), sex at birth (9%, female vs.~7%, male).

#block[
#figure([
#{set text(font: ("Arial Narrow", "Source Sans Pro", "sans-serif")); table(
  columns: 4,
  align: (left,left,left,left,),
  table.header(table.cell(align: left)[Contrast], table.cell(align: left)[Median], table.cell(align: left)[Bayesian CI], table.cell(align: left)[Probability],),
  table.hline(),
  table.cell(align: left)[Fairness Contrasts (No Lag)], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 4, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[female vs. male], table.cell(align: left)[-0.043], table.cell(align: left)[\[-0.059, -0.028\]], table.cell(align: left)[1],
  table.cell(align: left)[non-White and/or Hispanic vs. non-Hispanic White], table.cell(align: left)[-0.131], table.cell(align: left)[\[-0.222, -0.057\]], table.cell(align: left)[0.999],
  table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[below poverty line vs. above poverty line], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[-0.012], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[\[-0.033, 0.007\]], table.cell(align: left, stroke: (bottom: (paint: black, thickness: 0.75pt)))[0.848],
  table.cell(align: left)[Fairness Contrasts (2-week Lag)], table.cell(align: left)[], table.cell(align: left)[], table.cell(align: left)[],
  table.cell(colspan: 4, stroke: (bottom: (paint: black, thickness: 2.25pt)))[#strong[]],
  table.cell(align: left)[female vs. male], table.cell(align: left)[-0.098], table.cell(align: left)[\[-0.125, -0.073\]], table.cell(align: left)[1],
  table.cell(align: left)[non-White and/or Hispanic vs. non-Hispanic White], table.cell(align: left)[-0.13], table.cell(align: left)[\[-0.208, -0.058\]], table.cell(align: left)[0.998],
  table.cell(align: left)[below poverty line vs. above poverty line], table.cell(align: left)[-0.039], table.cell(align: left)[\[-0.073, -0.008\]], table.cell(align: left)[0.98],
  table.hline(),
  table.footer([#super[] Median auROC differences less than 0 indicate the model, on average, performed worse for the disadvantaged group (female, non-White and/or Hispanic, income below poverty line) compared to the advantaged group (male, non-Hispanic White, income above poverty line). Bayesian CI represents the range of values where there is a 95% probability that the true auROC difference lies within that range. Probability indicates the posterior probability that this difference is smaller than 0 (i.e., the models are performing differently for fairness subgroups).], [], [], [],),
)}
], caption: figure.caption(
position: top, 
[
Median difference in auROC, 95% Bayesian credible interval (CI), and posterior probability that that the auROC difference was smaller than 0 for fairness contrasts for the no lag and 2-week lagged models.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-fairness>


]
== Model Calibration
<model-calibration-1>
The raw probabilities produced by our final models were not well calibrated. Consequently, we used Platt scaling to improve calibration. Platt scaling showed excellent improvement to the no lag model with a Brier score of .043. Calibration also improved probability accuracy for the 2-week lagged model with a Brier score of .063. For comparison, raw probability scores yielded Brier scores of .071 and .077 for the no lag and 2-week lagged models, respectively.

#ref(<fig-cal>, supplement: [Figure]) shows the calibration plots for the raw and calibrated probabilities for the no lag and 2-week lagged model. It also includes a histogram of raw probabilities that demonstrates our models produced variable predicted probabilities, spanning nearly the entire 0 - 1 range. Calibration plots and Brier scores for all 5 models are available in the supplement.

#block[
#figure([
#box(image("index_files/figure-typst/notebooks-mak_figures-fig-cal-output-1.png"))
], caption: figure.caption(
position: bottom, 
[
Calibration plots of raw and calibrated lapse probabilities for the baseline (no lag) and 2-week lagged models. Predicted probabilities (x-axis) are binned into deciles. Observed lapse probability (y-axis) represents the proportion of actual lapses observed in each bin. The dashed diagonal represents perfect calibration. Points below the line indicate overestimation and points above the line indicate underestimation. Raw probabilities are depicted as solid black curves. Platt calibrated probabilities are depicted as pink dashed curves. The grey histogram along the bottom of the plot represents the proportion of raw probabilities in each bin.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-cal>


]
== Feature Importance
<feature-importance>
Global feature importance is an indicator of how important a feature category was to the model's predictions, on average (i.e., across all participants and all observations). The top globally important feature category (i.e., highest mean |Shapley value|) for all models was past use. Future efficacy was a strong predictor for more immediate model predictions (i.e., no lag), but its importance diminished as lag time increased. On the other hand, as lag time increased, past/future risky situations increased in importance. Craving was consistently important in magnitude across all models.

#ref(<fig-4>, supplement: [Figure]) shows the relative ranking of feature categories for the no lag and 2-week lagged models. Feature importance plots for all lag times is available in the supplement. These findings were also consistent across demographic subgroups (plots of feature importance by demographic group are available for the no lag and 2-week lagged models in the supplement).

#block[
#figure([
#box(image("index_files/figure-typst/notebooks-mak_figures-fig-4-output-1.png"))
], caption: figure.caption(
position: bottom, 
[
Global (mean |Shapley value|) and local (proportion of days as top feature) feature importance for feature categories for the no lag and 2-week lagged models. Feature categories are ordered by their aggregate global importance. The importance of each feature category for each model is displayed separately by color.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-4>


]
= Discussion
<discussion>
== Model Evaluation and Comparisons
<model-evaluation-and-comparisons>
All the models that we evaluated performed exceptionally well. The no lag model, which predicts the probability of an immediate (i.e., within 24 hours) lapse back to alcohol use, had a .91 median posterior probability for auROC. Our 2-week lagged model, which made the most distal predictions, had a .85 median posterior probability for auROC, suggesting lagged models can be used to shift a 24-hour prediction window meaningfully into the future.

Across models (no lag, 1 day, 3 days, 1 week, and 2 weeks), model performance systematically decreased as models predicted further into the future. All lagged models had lower performance compared to the no lag baseline model and to the preceding adjacent lag model. This is unsurprising given what we know about prediction and substance use. Many important relapse risk factors are fluctuating processes that can first emerge and/or change day-by-day, if not more frequently. As lag time increases, features become less proximal to the start of the prediction window. Still, we wish to emphasize that our lowest auROC (.85) is still quite good, and the benefit of advanced notice (i.e., 2 weeks) likely outweighs the modest cost to model performance.

Collectively, these results suggest we can achieve clinically meaningful performance up to two weeks out. Our rigorous resampling methods (grouped, nested, k-fold cross-validation) make us confident that these are valid estimates of how our models would perform with new individuals. Furthermore, it should be noted that both the no lag and 2-week lagged models can be combined in a complementary fashion that allows both for highly accurate immediate lapse prediction and advanced warning about future lapse risk.

== Model Fairness
<model-fairness>
In recent years, the machine learning field has begun to recognize the need to evaluate model fairness when algorithms are used to inform important decisions (e.g., healthcare services offered, eligibility for loans, early parole). Algorithms that perform favorably for only majority group members may exacerbate existing disparities in access to resources and important clinical outcomes \[40\]. In this study, we assessed model fairness by comparing model performance across important subgroups with known disparities in substance use treatment access and/or outcomes - race/ethnicity (non-White and/or Hispanic vs.~non-Hispanic White), income (below poverty line vs.~above poverty line), and sex at birth (female vs.~male).

All models performed worse for people who were non-White and/or Hispanic, and for people who had an income below the poverty line. The lack of diversity in our training data was likely a key contributor to the poorer model performance in these subgroups. Participants of color were severely underrepresented in our training data (#emph[N] = 20, 13%). Individuals below the poverty line were also underrepresented, though to a lesser degree (#emph[N] = 49, 32%). Of course, these model comparisons should be considered preliminary and interpreted cautiously because the low sample sizes in some groups (e.g., non-White and/or Hispanic) may reduce the statistical validity of the analyses. Nonetheless, these preliminary results regarding model fairness should motivate further efforts to evaluate and address differences in model performance across groups that differ in SUD treatment access/outcome disparities.

An obvious solution to this problem involves intentional recruitment for diversity in training data when developing prediction models. For example, we are now working to increase the racial, ethnic, and income diversity of our training data for alcohol lapse prediction while simultaneously optimizing feedback from these models for implementation purposes \[24\]. In a separate project, we developed a national recruitment method that enabled us to recruit for racial, ethnic and income diversity while also focusing on much needed diversity across geographic location (e.g., rural vs.~urban; \[19\]). We expect geographic diversity in the training data may also be crucial to develop fair models because the features that predict lapse in urban and suburban settings may differ from those that predict lapse in rural environments. If rural participants are not used to train models, the implementation of these models may compound existing disparities in SUD treatment in these communities \[41,42\].

Future research can also explore potential computational solutions to mitigate performance disparities that emerge when subgroups are poorly represented in available training data. For example, training data from under-represented subgroups could be up-sampled (e.g., using the synthetic minority oversampling technique), or the cost functions used by the learning algorithms could be adjusted to differentially weigh prediction errors based on participant characteristics. In another vein, modeling approaches that yield idiographic, person-specific models \[43--46\] may reduce performance disparities across subgroups. For example, we have begun to develop state space models whose parameters can be initialized with priors derived from existing training data but then adjusted over time to fit patterns present within a specific individual's time-series \[47\]. Such models may mitigate issues of unfairness to a large degree because they will weigh the individual's own data more heavily than group level estimates over time as more data accrue.

Of note, problems with model fairness can emerge even when subgroups are well-represented in the training data. Our models performed less well for women compared to men despite the fact that women were well-represented in the training data (#emph[N] = 74, 49%). Instead, this differential performance may have resulted from more fundamental problems with the features available to the model. We chose our EMA items using domain expertise from decades of research on the factors that predict relapse. However, prior to the 1993 National Institute of Health Revitalization Act \[48\] that mandated the inclusion of minorities and women in research, women were mostly excluded from substance use treatment research due to their childbearing potential \[49\]. As a result, it is possible that our theories about the causes and contributors to relapse are biased toward constructs that are more relevant for men than women. If true, features derived from EMA items that tap these constructs would be expected to under-perform when predicting lapses for women. More research may be needed to identify relapse risk factors for women (e.g., interpersonal relationship problems \[50\], hormonal changes \[51\]), and other groups under-represented in the literature before we can fully address these performance disparities.

In the meantime, data-driven (bottom-up) approaches can be used to engineer high-dimensional feature sets that are not explicitly grounded in existing, and potentially biased, theories. For example, we have begun to explore the application of natural language processing techniques (e.g., LIWC; topic modeling; BERT \[52--54\]) to text messages and other social media activity by our participants to engineer features that may predict future lapses. Such features may or may not align with existing theories about relapse, but because they are anchored to participants' own words, they may serve as reliable indicators of lapse risk for certain individuals, particularly when used within learning algorithms that employ feature selection, regularization, or other techniques to address the bias-variance trade-off with high-dimensional feature sets. Furthermore, emerging techniques for interpreting machine learning models \[55\] can be applied to models that perform well to bootstrap the identification of new lapse risk constructs based on these novel features.

Beyond issues of training data representation and lacunae or outright biases in our theories, it is also true that historically marginalized groups that have experienced systemic racism, exclusion, or other stigma around substance use (e.g., societal expectations for women regarding attractiveness, cleanliness and motherhood \[56\]) may feel less trusting in disclosing substance use \[57\]. These experiences could prompt some individuals in these subgroups to under-report lapses and/or risk factors, which could also degrade performance and evaluation of our models for these subgroups. We observed relatively comparable percentages of lapses reported among disadvantaged compared to advantaged groups. However, comparable lapse rates do not necessarily confirm comparable reporting accuracy because it is possible that there were systematic differences in lapse rates across groups that were masked by issues of trust.

== Model Characterization
<model-characterization>
=== Calibration
<calibration>
After applying Platt scaling to our predicted probabilities, our models were generally well calibrated with increasing monotonic relationships between calibrated model output and lapse event rates. Well-calibrated probabilities indicate that the predicted probability aligns closely with the true likelihood of an outcome (i.e., a lapse). Our no lag model had excellent calibration. However, the calibration plots suggest that with a longer lag time of 2 weeks, the model tends to over-predict the likelihood of lapses when predicted probabilities were higher.

This pattern may not necessarily be problematic. Research suggests that people often struggle to interpret probabilistic feedback, especially when it's provided in raw numerical form \[58--60\]. As a result, it may be more effective to communicate risk using coarser categories (e.g., low, medium, or high risk) or through relative changes in risk (e.g., “Your risk of lapse is higher this week compared to last week”). These forms of feedback may be less sensitive to small miscalibrations at the extremes as long as the relationship between predicted probabilities and the observed event rate is monotonic.

=== Feature importance
<feature-importance-1>
The relative ordering of top global features remained somewhat consistent across the no lag and 2-week lagged models. Past use was the most important feature in both models in our dataset. This is not surprising given that our outcome was lapse, and past behavior is often the best predictor of future behavior. This finding also supports decades of clinical research on relapse prevention, where lapses (i.e., single instances of goal inconsistent alcohol use) are seen as powerful precursors to relapse (i.e., full return to harmful drinking; \[8\]). Abstinence self-efficacy emerged as the second most important feature in both models in our dataset, indicating that participants had reasonably accurate insight into their near-term success with maintaining alcohol abstinence. Craving was also an important predictor in both models, suggesting that it may be an important target for intervention to support early recovery efforts.

Several feature categories displayed sizeable differences in global importance by lag time. The importance of abstinence self-efficacy dropped by more than 50% in the 2-week lagged model relative to the no lag model. This may indicate that self-efficacy during early recovery is unstable even across shorter periods of time such that their current self-efficacy does not strongly predict abstinence success even two weeks into the future. In fact, craving and risky situations become as important as self-efficacy when predicting two-week lagged lapses. It may be that these other experiences are shaping and changing the individual's self-efficacy rapidly in early recovery. This also suggests that more frequent clinical assessments of self-efficacy as a target for intervention may be needed rather than assuming stability in this construct if initial assessment suggests it is high. Also, our study cannot determine if this differential importance of self-efficacy for immediate vs.~lagged lapses persists beyond early recovery (where people may be encouraged to take a "one day at a time" mindset). Self-efficacy may become a more stable predictor of future abstinence success after longer periods of recovery, but our sample was limited to participants in early recovery (\<= 8 weeks of abstinence at intake).

Past use was less important for the 2-week lagged model compared to the no lag model. This indicates that the predictive strength of a lapse on the likelihood of subsequent lapses diminishes to some degree over a relatively short period of time. This is good news and reinforces that single lapses do not always mark a return to consistent patterns of frequent, and potentially harmful, alcohol use. Despite this reduction in importance of past use as a predictor of lagged alcohol use, past use did remain the most important category for two-week lagged lapses. Lapses may provide "teachable moments" that can be used to reinforce recovery motivation, better understand risks, and develop skills to address those risks \[10\]. Conversely, lapses should not be ignored because they remain strong predictors of further use.

Surprisingly, past and anticipated risky situations were more important in the 2-week lagged vs.~no lag model, suggesting that the impact of these situations on lapses back to use may be delayed. It may be that persistent exposure to risks is necessary to undermine an abstinence goal and lead to return to alcohol use. Alternatively or additionally, people may also be better able to anticipate future risky situations (e.g., vacations, anniversaries of significant dates) than future acute stressors or even future self-efficacy. Regardless, the increased importance of risky situations for predicting lagged lapses provides an opportunity to intervene prior to the lapse, particularly if the individual is encouraged to assess future risks and/or makes use of a recovery monitoring prediction model.

We were also surprised that stressful events, pleasant events, and affective state features did not make more important contributions to predictions across models. These constructs are highlighted in numerous theories about addiction and relapse \[8,9,61\] and represent targets for intervention in many existing treatments \[17,62--64\]. It may be that their impact is subsumed within other more powerful features (i.e., past use and self-efficacy). However, this seems unlikely given that the methodology underlying Shapley values allows for a fair distribution of importance among the relevant predictive features even when those features are correlated \[55\]. Alternatively, we may need to more carefully consider the nuanced roles that these constructs play (e.g., within the context of individual coping strategies, social support or environmental factors) in the return to alcohol use during recovery \[65\].

== Additional Limitations and Future Directions
<additional-limitations-and-future-directions>
We believe our lapse prediction models will be most effective when embedded in an RMSS designed to deliver adaptive and personalized continuing care. This system could send daily, weekly, or less frequent messages to users with personalized feedback about their risk of lapse and provide support recommendations tailored to their current recovery needs. This study provides initial support that immediate and lagged prediction models can be trained to high accuracy using EMA for recovery monitoring. Furthermore, locally important features from these models can be used to identify the specific factors that contribute to each lapse risk prediction.

The no lag model can be used to guide individuals to take immediate, actionable steps to maintain their recovery goals and support them to implement these steps within the RMSS. For example, the RMSS can recommend an embedded urge surfing activity when someone's immediate risk is driven by strong craving whereas a guided relaxation video can be provided to the user when they report stressful events. Similarly, the RMSS can encourage (and explicitly support) the user to reflect on recent past successes and/or skills they have developed when their self-efficacy is low.

The 2-week lagged model provides individuals with advanced warning of their lapse risk. This model is well-suited to support recovery needs that cannot be addressed immediately within an RMSS app, such as scheduling positive or pleasant activities, increasing social engagement, or attending a peer-led recovery meeting. To be clear, we do not believe an RMSS app alone will be sufficient to deliver continuing care. We expect individuals will require additional support throughout their recovery from a mental health provider (e.g., motivational enhancement, crisis management, skill building), a peer (e.g., sponsor, support group), or family member. Importantly, these types of supports take time to set up, highlighting the value of this lagged 2-week model.

At this point, it is still unclear the best way to provide risk and support information from our models to people. For an RMSS to be successful, users must trust the system, consistently engage with the system over time, and find the system beneficial. We have recently launched an NIAAA funded project to optimize daily support messages by examining the impact of several key message components (e.g., lapse probability, locally important features, a risk-relevant recovery activity recommendation, the linguistic style and tone of the message) on engagement, trust, clinical outcomes \[66\].

For a system using lagged models, we can imagine that lags longer than two weeks (i.e., more advanced warning) would be better still. In the present study, we could not train models with lags longer than two weeks because participants only provided lapse reports for up to three months. With the 2-week lagged model, we had approximately 17% fewer labeled observations for training because the first two weeks (out of 12 weeks) of labels for each participant were discarded. This data loss may be one factor that contributed to the decreases in model performance with increases in lag time and we believed that greater data loss (e.g., 25% for a 3-week lag) would not be tenable. We have recently completed data collection on a NIDA funded project where participants provided EMA and other sensed data for up to 12 months \[19\]. These data will allow us to train models with longer lags and to better evaluate the impact of data loss on model performance because lag time can be increased substantially with proportionally less data loss given 52 weeks of labeled observations per participant.

While the number of individual observations (i.e., \# prediction timepoints per participant X \# participants) was sufficient to train models with low bias, low variance and overall high performance, the relatively small number of unique participants remains an important limitation of this study. Successful machine learning models must generalize well to new data (i.e.~new observations among new individuals). Our analyses only speak to how well our models generalize to a small number of new individuals and these individuals are mostly homogeneous with respect to key demographic characteristics. To be clear, we made the most efficient use of our sample (i.e., using nested cross-validation to maximize the amount of held-out/test sets available for model evaluation); however, concerns about the generalizability to new individuals and other populations (e.g., individuals living in rural locations, members of other diverse groups not adequately represented in our sample) remain.

Our models predicted goal-inconsistent alcohol use. In our sample, in which all participants had a goal of abstinence, goal-inconsistent use was a homogeneous, observable behavior (i.e., any alcohol use). However, not all individuals in recovery choose complete abstinence. It remains an open question whether our predictors of goal-inconsistent use would generalize to alternative recovery goals (e.g., moderation). Future studies could allow individuals to self-define goal-inconsistent use (e.g., drinking on a weekday or drinking more than planned) and train models in samples with diverse recovery goals.

Additionally, our use of features from 4x daily EMA as model inputs may raise concerns about measurement burden. We confirmed that participants can comply with such EMA schedules over this time period and that they find it acceptable given its potential benefits to them \[18,see also 67\]. However, frequent daily surveys may become too burdensome within an RMSS intended for use over many, many months to years for long-term continuing care. We have begun to address this concern by training no lag models with fewer EMAs (1x daily) and have found comparable performance \[47\]. Additionally, reinforcement learning could potentially be used for adaptive EMA sampling. For example, each day the algorithm could make a decision to send out an EMA or not based on inferred latent states of the individual based on previous EMA responses and predicted probability of lapse.

We have also begun to explore how we can supplement our models with data from lower burden sensing methods. Geolocation, which can be passively sensed, could compliment EMA well \[68\]. First, it could provide insight into information not easily captured by self-report without lengthy surveys. For example, the amount of time spent in risky locations, or changes in routine (e.g., loss of job; move to new city) that could indicate life stressors can be detected in movement patterns. Second, the near-continuous sampling of geolocation could offer risk-relevant information that would otherwise be missed in between the discrete sampling periods of EMA. Furthermore, potentially powerful features can be engineered by combining geolocation data with contextual information available in public sources (e.g., census data, alcohol outlet density) \[69,70\] or collected from the user directly (e.g., self-evaluated riskiness of a given location) \[19\].

== Conclusions
<conclusions>
This study suggests it is possible to accurately predict alcohol lapses both immediately and up to two weeks into the future using lagged machine learning prediction models. The no lag model could guide users to engage with a smart RMSS that provides daily recovery activities that are personalized to their lapse risk and the factors contributing to that risk. The 2-week lagged model could enable patients to seek out and implement recovery support that is not immediately available to them within the RMSS. Several important steps remain prior to implementing the no lag and 2-week lagged models within a smart RMSS. Feedback and support messages from these models should be optimized to sustain system engagement, trust, and clinical outcomes. Passive sensing of model inputs may allow assessment of a broader range of risk factors with less burden for system users. Perhaps most important, model fairness must be improved by decreasing disparities in performance for less privileged groups. We remain optimistic about the potential to implement these models within a smart RMSS because these barriers, while challenging, are surmountable.

== Acknowledgments
<acknowledgments>
This research was supported by grants from the National Institute on Alcohol Abuse and Alcoholism (NIAAA; R01 AA024391 to John J. Curtin) and the National Institute on Drug Abuse (NIDA; R01 DA047315 to John J. Curtin). The authors wish to thank Susan E. Wanta for her role as the project administrator.

= References
<references>
#block[
#block[
\1. McLellan AT, Lewis DC, O'Brien CP, Kleber HD. Drug dependence, a chronic medical illness: Implications for treatment, insurance, and outcomes evaluation. JAMA. 2000;284: 1689--1695. doi:#link("https://doi.org/10.1001/jama.284.13.1689")[10.1001/jama.284.13.1689]

] <ref-mclellanDrugDependenceChronic2000>
#block[
\2. Dennis M, Scott CK. #link("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2797101")[Managing Addiction as a Chronic Condition]. Addiction Science & Clinical Practice. 2007;4: 45--55.

] <ref-dennisManagingAddictionChronic2007>
#block[
\3. Hedegaard H, Miniño AM, Spencer MR, Warner M. Drug overdose deaths in the United States, 1999--2020. 2021.

] <ref-hedegaardDrugOverdoseDeaths2021>
#block[
\4. Centers for Disease Control and Prevention (CDC). Annual Average for United States 2011--2015 Alcohol-Attributable Deaths Due to Excessive Alcohol Use, All Ages. 2022 Alcohol Related Disease Impact (ARDI) Application Website. https:\/\/nccd.cdc.gov/DPH\_ARDI/Default/Default.aspx;

] <ref-centersfordiseasecontrolandpreventioncdcAnnualAverageUnited>
#block[
\5. Wagner EH, Austin BT, Davis C, Hindmarsh M, Schaefer J, Bonomi A. Improving Chronic Illness Care: Translating Evidence Into Action. Health Affairs. 2001;20: 64--78. doi:#link("https://doi.org/10.1377/hlthaff.20.6.64")[10.1377/hlthaff.20.6.64]

] <ref-wagnerImprovingChronicIllness2001>
#block[
\6. Stanojlović M, Davidson L. Targeting the Barriers in the Substance Use Disorder Continuum of Care With Peer Recovery Support. Substance Abuse: Research and Treatment. 2021;15: 1178221820976988. doi:#link("https://doi.org/10.1177/1178221820976988")[10.1177/1178221820976988]

] <ref-stanojlovicTargetingBarriersSubstance2021>
#block[
\7. Socías ME, Volkow N, Wood E. Adopting the “cascade of care” framework: An opportunity to close the implementation gap in addiction care? Addiction. 2016;111: 2079--2081. doi:#link("https://doi.org/10.1111/add.13479")[10.1111/add.13479]

] <ref-sociasAdoptingCascadeCare2016>
#block[
\8. Marlatt GA, Gordon JR, editors. Relapse Prevention: Maintenance Strategies in the Treatment of Addictive Behaviors. First edition. New York: The Guilford Press; 1985.

] <ref-marlattRelapsePreventionMaintenance1985>
#block[
\9. Witkiewitz K, Marlatt GA. Relapse prevention for alcohol and drug problems: That was zen, this is tao. American Psychologist. 2004;59: 224--235. doi:#link("https://doi.org/10.1037/0003-066X.59.4.224")[10.1037/0003-066X.59.4.224]

] <ref-witkiewitzRelapsePreventionAlcohol2004>
#block[
\10. Witkiewitz K, Marlatt GA. Modeling the complexity of post-treatment drinking: It's a rocky road to relapse. Clinical Psychology Review. 2007;27: 724--738. doi:#link("https://doi.org/10.1016/j.cpr.2007.01.002")[10.1016/j.cpr.2007.01.002]

] <ref-witkiewitzModelingComplexityPosttreatment2007>
#block[
\11. Brandon TH, Vidrine JI, Litvin EB. Relapse and relapse prevention. Annual Review of Clinical Psychology. 2007;3: 257--284. doi:#link("https://doi.org/10.1146/annurev.clinpsy.3.022806.091455")[10.1146/annurev.clinpsy.3.022806.091455]

] <ref-brandonRelapseRelapsePrevention2007>
#block[
\12. Bickman L, Lyon AR, Wolpert M. Achieving Precision Mental Health through Effective Assessment, Monitoring, and Feedback Processes. Administration and Policy in Mental Health and Mental Health Services Research. 2016;43: 271--276. doi:#link("https://doi.org/10.1007/s10488-016-0718-5")[10.1007/s10488-016-0718-5]

] <ref-bickmanAchievingPrecisionMental2016>
#block[
\13. DeRubeis RJ. The history, current status, and possible future of precision mental health. Behaviour Research and Therapy. 2019;123: 103506. doi:#link("https://doi.org/10.1016/j.brat.2019.103506")[10.1016/j.brat.2019.103506]

] <ref-derubeisHistoryCurrentStatus2019>
#block[
\14. Kranzler HR, McKay JR. Personalized Treatment of Alcohol Dependence. Current Psychiatry Reports. 2012;14: 486--493. doi:#link("https://doi.org/10.1007/s11920-012-0296-5")[10.1007/s11920-012-0296-5]

] <ref-kranzlerPersonalizedTreatmentAlcohol2012>
#block[
\15. Mohr DC, Zhang M, Schueller SM. Personal Sensing: Understanding Mental Health Using Ubiquitous Sensors and Machine Learning. Annual Review of Clinical Psychology. 2017;13: 23--47. doi:#link("https://doi.org/10.1146/annurev-clinpsy-032816-044949")[10.1146/annurev-clinpsy-032816-044949]

] <ref-mohrPersonalSensingUnderstanding2017>
#block[
\16. Hastie T, Tibshirani R, Friedman JH. The elements of statistical learning: Data mining, inference, and prediction. 2nd ed. New York, NY: Springer; 2009.

] <ref-hastieElementsStatisticalLearning2009>
#block[
\17. Bowen S, Chawla N, Grow J, Marlatt GA. Mindfulness-Based Relapse Prevention for Addictive Behaviors: A Clinician's Guide. Second edition. New York: The Guilford Press; 2021.

] <ref-bowenMindfulnessBasedRelapsePrevention2021>
#block[
\18. Wyant K, Moshontz H, Ward SB, Fronk GE, Curtin JJ. Acceptability of Personal Sensing Among People With Alcohol Use Disorder: Observational Study. JMIR mHealth and uHealth. 2023;11: e41833. doi:#link("https://doi.org/10.2196/41833")[10.2196/41833]

] <ref-wyantAcceptabilityPersonalSensing2023>
#block[
\19. Moshontz H, Colmenares AJ, Fronk GE, Sant'Ana SJ, Wyant K, Wanta SE, et al. Prospective Prediction of Lapses in Opioid Use Disorder: Protocol for a Personal Sensing Study. JMIR Research Protocols. 2021;10: e29563. doi:#link("https://doi.org/10.2196/29563")[10.2196/29563]

] <ref-moshontzProspectivePredictionLapses2021>
#block[
\20. Wyant K, Sant'Ana SJK, Fronk G, Curtin JJ. Machine learning models for temporally precise lapse prediction in alcohol use disorder. Psychopathology and Clinical Science. 2024. doi:#link("https://doi.org/10.31234/osf.io/cgsf7")[10.31234/osf.io/cgsf7]

] <ref-wyantMachineLearningModels2024>
#block[
\21. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. Proceedings of the 31st International Conference on Neural Information Processing Systems. Red Hook, NY, USA: Curran Associates Inc.; 2017. pp. 4768--4777.

] <ref-lundbergUnifiedApproachInterpreting2017>
#block[
\22. Soyster PD, Ashlock L, Fisher AJ. Pooled and person-specific machine learning models for predicting future alcohol consumption, craving, and wanting to drink: A demonstration of parallel utility. Psychology of Addictive Behaviors: Journal of the Society of Psychologists in Addictive Behaviors. 2022;36: 296--306. doi:#link("https://doi.org/10.1037/adb0000666")[10.1037/adb0000666]

] <ref-soysterPooledPersonspecificMachine2022>
#block[
\23. Walters ST, Businelle MS, Suchting R, Li X, Hébert ET, Mun E-Y. Using machine learning to identify predictors of imminent drinking and create tailored messages for at-risk drinkers experiencing homelessness. Journal of Substance Abuse Treatment. 2021;127: 108417. doi:#link("https://doi.org/10.1016/j.jsat.2021.108417")[10.1016/j.jsat.2021.108417]

] <ref-waltersUsingMachineLearning2021>
#block[
\24. Wyant K, Sant'Ana SJ, Punturieri CE, Yu J, Fronk GE, Maggard CM, et al. Maximizing engagement, trust, and clinical benefit of AI-generated recovery monitoring and support messages for alcohol use disorder: Protocol for an optimization study. under review.

] <ref-wyantMaximizingEngagementTrustunderreview>
#block[
\25. Pinedo M. A current re-examination of racial/ethnic disparities in the use of substance abuse treatment: Do disparities persist? Drug and Alcohol Dependence. 2019;202: 162--167. doi:#link("https://doi.org/10.1016/j.drugalcdep.2019.05.017")[10.1016/j.drugalcdep.2019.05.017]

] <ref-pinedoCurrentReexaminationRacial2019>
#block[
\26. Kilaru AS, Xiong A, Lowenstein M, Meisel ZF, Perrone J, Khatri U, et al. Incidence of Treatment for Opioid Use Disorder Following Nonfatal Overdose in Commercially Insured Patients. JAMA Network Open. 2020;3: e205852. doi:#link("https://doi.org/10.1001/jamanetworkopen.2020.5852")[10.1001/jamanetworkopen.2020.5852]

] <ref-kilaruIncidenceTreatmentOpioid2020>
#block[
\27. Olfson M, Mauro C, Wall MM, Choi CJ, Barry CL, Mojtabai R. Healthcare coverage and service access for low-income adults with substance use disorders. Journal of Substance Abuse Treatment. 2022;137: 108710. doi:#link("https://doi.org/10.1016/j.jsat.2021.108710")[10.1016/j.jsat.2021.108710]

] <ref-olfsonHealthcareCoverageService2022>
#block[
\28. Greenfield SF, Brooks AJ, Gordon SM, Green CA, Kropp F, McHugh RK, et al. Substance abuse treatment entry, retention, and outcome in women: A review of the literature. Drug and Alcohol Dependence. 2007;86: 1--21. doi:#link("https://doi.org/10.1016/j.drugalcdep.2006.05.012")[10.1016/j.drugalcdep.2006.05.012]

] <ref-greenfieldSubstanceAbuseTreatment2007>
#block[
\29. Hsieh F. Sample size tables for logistic regression. Statistics in Medicine. 1989;8: 795--802.

] <ref-hsiehSampleSizeTables1989>
#block[
\30. Derogatis, L.R. Brief Symptom Inventory 18 - Administration, scoring, and procedures manual. Minneapolis: NCS Pearson; 2000.

] <ref-derogatislBriefSymptomInventory>
#block[
\31. WHO ASSIST Working Group. #link("https://www.ncbi.nlm.nih.gov/pubmed/12199834")[The Alcohol, Smoking and Substance Involvement Screening Test (ASSIST): Development, reliability and feasibility]. Addiction (Abingdon, England). 2002;97: 1183--1194.

] <ref-whoassistworkinggroupAlcoholSmokingSubstance2002>
#block[
\32. Kuhn M, Wickham H. Tidymodels: A collection of packages for modeling and machine learning using tidyverse principles. 2020.

] <ref-kuhnTidymodelsCollectionPackages2020>
#block[
\33. Kuhn M. Tidyposterior: Bayesian Analysis to Compare Models using Resampling Statistics. 2022.

] <ref-kuhnTidyposteriorBayesianAnalysis2022>
#block[
\34. Goodrich B, Gabry J, Ali I, Brilleman S. Rstanarm: Bayesian Applied Regression Modeling via Stan. 2023.

] <ref-goodrichRstanarmBayesianApplied2023>
#block[
\35. Center for High Throughput Computing. Center for high throughput computing. Center for High Throughput Computing; 2006. doi:#link("https://doi.org/10.21231/GNT1-HW21")[10.21231/GNT1-HW21]

] <ref-chtc>
#block[
\36. Kuhn M, Johnson K. Applied Predictive Modeling. 1st ed. 2013, Corr. 2nd printing 2018 edition. New York: Springer; 2018. doi:#link("https://doi.org/10.1007/978-1-4614-6849-3")[10.1007/978-1-4614-6849-3]

] <ref-kuhnAppliedPredictiveModeling2018>
#block[
\37. RStudio Team. RStudio: Integrated Development for R. Boston, MA: RStudio, Inc; 2020.

] <ref-rstudioteamRStudioIntegratedDevelopment2020>
#block[
\38. Gabry J, Goodrich B. Prior Distributions for rstanarm Models. CRAN R-Project. https:\/\/cran.r-project.org/web/packages/rstanarm/vignettes/priors.html; 2023.

] <ref-gabryPriorDistributionsRstanarm2023>
#block[
\39. Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers. MIT Press; 1999. pp. 61--74.

] <ref-plattProbabilisticOutputsSupport1999>
#block[
\40. Veinot TC, Mitchell H, Ancker JS. Good intentions are not enough: How informatics interventions can worsen inequality. Journal of the American Medical Informatics Association: JAMIA. 2018;25: 1080--1088. doi:#link("https://doi.org/10.1093/jamia/ocy052")[10.1093/jamia/ocy052]

] <ref-veinotGoodIntentionsAre2018>
#block[
\41. Lee JH, Wheeler DC, Zimmerman EB, Hines AL, Chapman DA. Urban--Rural Disparities in Deaths of Despair: A County-Level Analysis 2004--2016 in the U.S. American journal of preventive medicine. 2023;64: 149--156. doi:#link("https://doi.org/10.1016/j.amepre.2022.08.022")[10.1016/j.amepre.2022.08.022]

] <ref-leeUrbanRuralDisparities2023>
#block[
\42. Lister JJ, Weaver A, Ellis JD, Himle JA, Ledgerwood DM. A systematic review of rural-specific barriers to medication treatment for opioid use disorder in the United States. The American Journal of Drug and Alcohol Abuse. 2020;46: 273--288. doi:#link("https://doi.org/10.1080/00952990.2019.1694536")[10.1080/00952990.2019.1694536]

] <ref-listerSystematicReviewRuralspecific2020>
#block[
\43. Fisher AJ. Toward a dynamic model of psychological assessment: Implications for personalized care. Journal of Consulting and Clinical Psychology. 2015;83: 825--836. doi:#link("https://doi.org/10.1037/ccp0000026")[10.1037/ccp0000026]

] <ref-fisherDynamicModelPsychological2015>
#block[
\44. David SJ, Marshall AJ, Evanovich EK, Mumma GH. Intraindividual Dynamic Network Analysis - Implications for Clinical Assessment. Journal of Psychopathology and Behavioral Assessment. 2018;40: 235--248. doi:#link("https://doi.org/10.1007/s10862-017-9632-8")[10.1007/s10862-017-9632-8]

] <ref-davidIntraindividualDynamicNetwork2018>
#block[
\45. Roche MJ, Pincus AL, Rebar AL, Conroy DE, Ram N. Enriching Psychological Assessment Using a Person-Specific Analysis of Interpersonal Processes in Daily Life. Assessment. 2014;21: 515--528. doi:#link("https://doi.org/10.1177/1073191114540320")[10.1177/1073191114540320]

] <ref-rocheEnrichingPsychologicalAssessment2014>
#block[
\46. Wright AGC, Hallquist MN, Stepp SD, Scott LN, Beeney JE, Lazarus SA, et al. Modeling Heterogeneity in Momentary Interpersonal and Affective Dynamic Processes in Borderline Personality Disorder. Assessment. 2016;23: 484--495. doi:#link("https://doi.org/10.1177/1073191116653829")[10.1177/1073191116653829]

] <ref-wrightModelingHeterogeneityMomentary2016>
#block[
\47. Pulick E, Curtin J, Mintz Y. Idiographic Lapse Prediction With State Space Modeling: Algorithm Development and Validation Study. JMIR Formative Research. 2025;9: e73265. doi:#link("https://doi.org/10.2196/73265")[10.2196/73265]

] <ref-pulickIdiographicLapsePrediction2025>
#block[
\48. Studies I of M(US)C on E and LIR to the I of W in C, Mastroianni AC, Faden R, Federman D. NIH Revitalization Act of 1993 Public Law 103-43. Women and Health Research: Ethical and Legal Issues of Including Women in Clinical Studies: Volume I. National Academies Press (US); 1994.

] <ref-studiesNIHRevitalizationAct1994>
#block[
\49. Vannicelli M, Nash L. Effect of Sex Bias on Women's Studies on Alcoholism. Alcoholism: Clinical and Experimental Research. 1984;8: 334--336. doi:#link("https://doi.org/10.1111/j.1530-0277.1984.tb05523.x")[10.1111/j.1530-0277.1984.tb05523.x]

] <ref-vannicelliEffectSexBias1984>
#block[
\50. Walitzer KS, Dearing RL. Gender differences in alcohol and substance use relapse. Clinical Psychology Review. 2006;26: 128--148. doi:#link("https://doi.org/10.1016/j.cpr.2005.11.003")[10.1016/j.cpr.2005.11.003]

] <ref-walitzerGenderDifferencesAlcohol2006>
#block[
\51. McHugh RK, Votaw VR, Sugarman DE, Greenfield SF. Sex and gender differences in substance use disorders. Clinical Psychology Review. 2018;66: 12--23. doi:#link("https://doi.org/10.1016/j.cpr.2017.10.012")[10.1016/j.cpr.2017.10.012]

] <ref-mchughSexGenderDifferences2018>
#block[
\52. Tausczik YR, Pennebaker JW. The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods. Journal of Language and Social Psychology. 2010;29: 24--54. doi:#link("https://doi.org/10.1177/0261927X09351676")[10.1177/0261927X09351676]

] <ref-tausczikPsychologicalMeaningWords2010>
#block[
\53. Blei DM, Ng AY, Jordan MI. Latent dirichlet allocation. J Mach Learn Res. 2003;3: 993--1022.

] <ref-bleiLatentDirichletAllocation2003>
#block[
\54. Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv; 2019. doi:#link("https://doi.org/10.48550/arXiv.1810.04805")[10.48550/arXiv.1810.04805]

] <ref-devlinBERTPretrainingDeep2019>
#block[
\55. Molnar C. Interpretable Machine Learning: A Guide For Making Black Box Models Explainable. Munich, Germany: Independently published; 2022.

] <ref-molnarInterpretableMachineLearning2022>
#block[
\56. Meyers SA, Earnshaw VA, D'Ambrosio B, Courchesne N, Werb D, Smith LR. The intersection of gender and drug use-related stigma: A mixed methods systematic review and synthesis of the literature. Drug and Alcohol Dependence. 2021;223: 108706. doi:#link("https://doi.org/10.1016/j.drugalcdep.2021.108706")[10.1016/j.drugalcdep.2021.108706]

] <ref-meyersIntersectionGenderDrug2021>
#block[
\57. Marwick AE, Boyd D. Privacy at the Margins Understanding Privacy at the Margins---Introduction. International Journal of Communication. 2018;12: 9.

] <ref-marwickPrivacyMarginsUnderstanding2018>
#block[
\58. Zikmund-Fisher BJ. The right tool is what they need, not what we have: A taxonomy of appropriate levels of precision in patient risk communication. Medical care research and review: MCRR. 2013;70: 37S--49S. doi:#link("https://doi.org/10.1177/1077558712458541")[10.1177/1077558712458541]

] <ref-zikmund-fisherRightToolWhat2013>
#block[
\59. Fagerlin A, Ubel PA, Smith DM, Zikmund-Fisher BJ. Making numbers matter: Present and future research in risk communication. American Journal of Health Behavior. 2007;31 Suppl 1: S47--56. doi:#link("https://doi.org/10.5555/ajhb.2007.31.supp.S47")[10.5555/ajhb.2007.31.supp.S47]

] <ref-fagerlinMakingNumbersMatter2007>
#block[
\60. Zipkin DA, Umscheid CA, Keating NL, Allen E, Aung K, Beyth R, et al. Evidence-based risk communication: A systematic review. Annals of Internal Medicine. 2014;161: 270--280. doi:#link("https://doi.org/10.7326/M14-0295")[10.7326/M14-0295]

] <ref-zipkinEvidencebasedRiskCommunication2014>
#block[
\61. Rawson RA, Shoptaw SJ, Obert JL, McCann MJ, Hasson AL, Marinelli-Casey PJ, et al. An intensive outpatient approach for cocaine abuse treatment. The Matrix model. Journal of Substance Abuse Treatment. 1995;12: 117--127. doi:#link("https://doi.org/10.1016/0740-5472(94)00080-b")[10.1016/0740-5472(94)00080-b]

] <ref-rawsonIntensiveOutpatientApproach1995>
#block[
\62. McHugh RK, Hearon BA, Otto MW. Cognitive-Behavioral Therapy for Substance Use Disorders. The Psychiatric clinics of North America. 2010;33: 511--525. doi:#link("https://doi.org/10.1016/j.psc.2010.04.012")[10.1016/j.psc.2010.04.012]

] <ref-mchughCognitiveBehavioralTherapySubstance2010>
#block[
\63. Liese BS, Beck AT. Cognitive-Behavioral Therapy of Addictive Disorders. First edition. New York: The Guilford Press; 2022.

] <ref-lieseCognitiveBehavioralTherapyAddictive2022>
#block[
\64. Center for Substance Abuse Treatment. Counselor's Treatment Manual: Matrix Intensive Outpatient Treatment for People With Stimulant Use Disorders. Rockville, MD: Substance Abuse and Mental Health Services Administration\; 2006.

] <ref-centerforsubstanceabusetreatmentCounselorsTreatmentManual2006>
#block[
\65. Fronk GE, Sant'Ana SJ, Kaye JT, Curtin JJ. Stress Allostasis in Substance Use Disorders: Promise, Progress, and Emerging Priorities in Clinical Research. Annual Review of Clinical Psychology. 2020;16: 401--430. doi:#link("https://doi.org/10.1146/annurev-clinpsy-102419-125016")[10.1146/annurev-clinpsy-102419-125016]

] <ref-fronkStressAllostasisSubstance2020>
#block[
\66. Wyant K, Sant'Ana SJ, Punturieri CE, Yu J, Fronk GE, Maggard CM, et al. Maximizing Engagement, Trust, and Clinical Benefit of AI-Generated Recovery Support Messages for Alcohol Use Disorder: Protocol for an Optimization Study. JMIR Research Protocols. 2025;14: e81697. doi:#link("https://doi.org/10.2196/81697")[10.2196/81697]

] <ref-wyantMaximizingEngagementTrust2025>
#block[
\67. Jones A, Remmerswaal D, Verveer I, Robinson E, Franken IHA, Wen CKF, et al. Compliance with ecological momentary assessment protocols in substance users: A meta-analysis. Addiction (Abingdon, England). 2019;114: 609--619. doi:#link("https://doi.org/10/gfsjzg")[10/gfsjzg]

] <ref-jonesComplianceEcologicalMomentary2019>
#block[
\68. Bae SW, Suffoletto B, Zhang T, Chung T, Ozolcer M, Islam MR, et al. Leveraging Mobile Phone Sensors, Machine Learning and Explainable Artificial Intelligence to Predict Imminent Same-Day Binge Drinking Events to Support Just-In-Time Adaptive Interventions: A Feasibility Study. JMIR formative research. 2023. doi:#link("https://doi.org/10.2196/39862")[10.2196/39862]

] <ref-baeLeveragingMobilePhone2023>
#block[
\69. Huang L, Li Q, Yue Y. Activity identification from GPS trajectories using spatial temporal POIs' attractiveness. Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Location Based Social Networks. New York, NY, USA: Association for Computing Machinery; 2010. pp. 27--30. doi:#link("https://doi.org/10.1145/1867699.1867704")[10.1145/1867699.1867704]

] <ref-huangActivityIdentificationGPS2010>
#block[
\70. Xie K, Deng K, Zhou X. From trajectories to activities: A spatio-temporal join approach. Proceedings of the 2009 International Workshop on Location Based Social Networks. New York, NY, USA: Association for Computing Machinery; 2009. pp. 25--32. doi:#link("https://doi.org/10.1145/1629890.1629897")[10.1145/1629890.1629897]

] <ref-xieTrajectoriesActivitiesSpatiotemporal2009>
] <refs>



