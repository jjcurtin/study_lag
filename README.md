A Quarto Manuscript Template

This is a template for generating a repo for a Quarto project for a study that will include

- a study manuscript
- notebooks that display relevant calculations/code/analyses presented in the manuscript
- poster(s) based on the study
- slide set(s) for talks based on the study


It is based off of a tutorial at: [Quarto Manuscripts: RStudio](https://quarto.org/docs/manuscripts/authoring/rstudio.html) but expanded to include linked posters and talks and with some customizations for our workflow and preferences.

## Introductory Reading

We strongly recommend that you read the following pages from the Quarto website to better understand their manuscript format

1. [Quarto Manuscripts](https://quarto.org/docs/manuscripts/)
2. [Authoring Manuscripts](https://quarto.org/docs/manuscripts/authoring/rstudio.html)
3. [Publishing Basics](https://quarto.org/docs/publishing/)
4. [Github Pages](https://quarto.org/docs/publishing/github-pages.html)
5. [Using Manuscripts](https://quarto.org/docs/manuscripts/components.html)

If you plan to make slide sets, you can also read docs on making slides in quarto generally and more specifically with the revealjs format

1. [Quarto Presentations](https://quarto.org/docs/presentations/) 
2. [Revealjs](https://quarto.org/docs/presentations/revealjs/)
3. [YAML formatting options](https://quarto.org/docs/reference/formats/presentations/revealjs.html)


## Getting Started

1. Go to the code tab for this repo
2. Use the green `Use this template` drop down and select `create new repository`
3. Name this repository with a name selected for your paper
4. Clone that new repository to your local computer
5. Rename the RStudio project file `study_template.Rproj` to the same name that you used for your repo (i.e., `study_[insert_name]`.Rproj`
6.  Begin working on your new manuscript!  
  - The main manuscript is saved as index.qmd.  
  - Additional quarto notebooks are saved in the /notebooks folder
  - qmd files for posters are saved individual folders within _presentations folder
  - qmd files for slide sets for talks are saved in individual folders within _presentations 


## Publishing Your Manuscript 


Github is a good choice to publish your manuscript and associated code from notebooks.  Your code is already hosted there so why not keep the manuscript website there as well?   There are [three different methods](https://quarto.org/docs/publishing/github-pages.html#publish-command) available to publish there.  We prefer the docs folder method. To use this method: 

1. You need to make sure the repo is public.  Go to Settings | Danger Zone and change visibility to public.

2. You need to configure GitHub to publish the website from the docs folder.  Go to Settings | Pages | Build and Deployment | Branch.  Set Branch to `main` and folder to `/docs` and save these settings.

3. In the terminal, go to the root of the folder and enter `touch .nojekyll` (this file should already be in your repo if you started from the template)

4. Render the manuscript.   This can be done from terminal using `quarto render` at the root of the quarto project  

5.  Add and commit the changes and push to Github.  The associated website will update automatically.



## Publishing Your Slideset

A Github repo can only have one website associated with it. If we are using the repo for the manuscript website, we need another location to publish slidesets from talks.  We have been using [Quarto Pub](https://quartopub.com/).   You can publish there using `quarto publish quarto-pub` [filename.qmd] from the folder that contains the qmd file for the talk.   The first time you do this, you will need to verify your account and the page address.  Follow the prompts.  This info is saved in _publish.yml so that you wont need to update it the next time you publish updates to your project.

More details are [here](https://quarto.org/docs/publishing/quarto-pub.html)
