%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nom : rapport_ienac.cls
% Date : 07/05/2018
% Auteur : Alexandre Chabory
%
% Classe pour les rapports étudiants IENAC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{FichiersSource/rapport_ienac}[07/05/2018 Classe pour projet de synthese IENAC, V1.0]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classe de base
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass[11pt,a4paper,french]{book}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\RequirePackage[square,sort,comma,numbers]{natbib} 

\RequirePackage[french]{babel}

% polices 
\RequirePackage{cmlgc}
\RequirePackage{ucs}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

% pour l'abstract
\RequirePackage{lipsum}

% figures
\RequirePackage{graphicx}
\RequirePackage{psfrag}
\RequirePackage{subfig}
\RequirePackage[usenames,dvipsnames]{pstricks}

% pour la commande \FloatBarrier
\RequirePackage{placeins}		    

\RequirePackage{enumitem}


% pour les tableaux.
\RequirePackage{tabularx} 

% pour entête et pied de page.
\RequirePackage{fancyhdr}           

% equations (caractères speciaux, ...)
\RequirePackage{amsfonts}               
\RequirePackage{amsgen}
\RequirePackage{amsbsy}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{bbm}
\RequirePackage{esint}				

% pour réduire les marges
\RequirePackage{fullpage} 

% Filigramme 
\RequirePackage{eso-pic}

% Acronyme
\RequirePackage[printonlyused]{acronym} % pour uniquement afficher ceux utilisés
%\RequirePackage[]{acronym}

% pour modifier l'interligne
\RequirePackage{setspace} 


\RequirePackage{float}  %Pour forcer les figures en position

\RequirePackage[pdftex,bookmarks,colorlinks,breaklinks]{hyperref} %Pour sommaire interactif
\RequirePackage{subfiles}
\DeclareGraphicsExtensions{.pdf,.png}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% definition de la mise en page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\voffset}        {-1.0cm}
\setlength{\topmargin}      {0cm} % Marge supérieure
\setlength{\headheight}     { 0.5cm} % Entête de page
%\setlength{\headwidth}     { 20cm} % Entête de page
\setlength{\headsep}        { 1cm} % Distance entre entête/début texte
%\setlength{\headwidth}      {14.0cm} % Distance entre entête/début texte
\setlength{\textheight}     {24.0cm} % Hauteur de la page texte
\setlength{\footskip}       { 1.5cm} % Pied de page (dans -> textheight)
\setlength{\oddsidemargin}  { 0cm} % Décalage page impair
\setlength{\evensidemargin} { 0cm} % Décalage page pair
\setlength{\textwidth}      {16cm} % Largeur de page
\setlength{\leftmargin}     { 0.5cm} % Décalage des item (par la gauche)
\setlength{\leftmargini}    { 0.5cm} % Décalage des item (par la gauche)
\setlength{\leftmarginii}   { 0.5cm} % Décalage des item (par la gauche)
\setlength{\rightmargin}    { 0.0cm} % Décalage des item (par la droite)
\setlength{\parskip}        { 0.2cm} % écart suplémentaire entre parag.
\setlength{\topsep}         { 0.0cm} % écart supl. entre parag. et item
\setlength{\partopsep}      { 0.0cm} % écart supl. entre parag. et item
\setlength{\itemsep}        { 0.0cm} % écart suplémentaire entre item
\setlength{\parsep}         { 0.0cm} % écart supl. entre parag d'un item
\setlength{\listparindent}  { 0.5cm} % Indentation dans les item
\setlength{\parindent}      { 0.cm} % Indentation
\newlength{\minindent}               % Indentation déclaration
\setlength{\minindent}      { 0.cm} % Indentation dans une minipage
\setitemize[0]{label=\textbullet} % Avoir des bullets points points ronds



\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}

% Changement de mode de numérotation des sous sous sections
\renewcommand{\thesubsubsection}{\thesubsection .\arabic{subsubsection}}

% pour resserrer un peu la table des matières
\addtocontents{toc}{
	\protect\setlength{\baselineskip}{1em}
	\protect\setlength{\parskip}{0pt}}


\tolerance 1414
\hbadness 1414
\emergencystretch 1.5em
\hfuzz 0.3pt
\widowpenalty=10000
\vfuzz \hfuzz
\raggedbottom

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% definition de la page de garde
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\supervisor#1{\gdef\@supervisor{#1}}
\def\maitrestage#1{\gdef\@maitrestage{#1}}
\def\typedoc#1{\gdef\@typedoc{#1}}
\def\promotion#1{\gdef\@promotion{#1}}
\def\datepassage#1{\gdef\@datepassage{#1}}

\renewcommand{\maketitle}{%
	\thispagestyle{empty}
	
	% Elargir pied de page
	\enlargethispage{2cm}
	% Logo et adresse ENAC en haut
	% \begin{minipage}{0.26\linewidth}
	% 	\includegraphics[width=2.5cm]{FichiersSource/Logos/logo_enac.eps} 
	% \end{minipage}
	\begin{minipage}{0.7\linewidth}
	\end{minipage}
	
	\rule{\textwidth}{0.8pt}\\[20pt]
	\begin{center} 
		% Type de document
		{\huge 
			\textbf{\@typedoc}\\[20pt]
			%  \large Version du : \today \\[30pt]
			
			
			% Titre du document
			\begin{minipage}{0.75\linewidth}
				\begin{center}
					\rule{\textwidth}{0.8pt}\\
					\begin{spacing}{1.0}
						{\Large \bf \@title} 
					\end{spacing}
					\rule{\textwidth}{0.8pt}\\[50pt]
				\end{center}
			\end{minipage}
			
			%~\\[80pt]
			\vfill
			
			% Date, auteurs, et encadrants
			\large {\@datepassage} \\[20pt]
			{\bf \@author}}\\[20pt]
		Supervis\'e par  \\[20pt]
		\large {\bf \@supervisor} \\[115pt]
	\end{center}
	
	\vfill
	
	\rule{\linewidth}{.5pt}
	% \begin{minipage}{0.15\linewidth}
	% 	\includegraphics[width=0.8\linewidth]{FichiersSource/Logos/logo_enac.eps} 
	% \end{minipage}
}   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tete de page et pied de page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.4pt}% Default \headrulewidth is 0.4pt
\renewcommand{\footrulewidth}{0.4pt}% Default \footrulewidth is 0pt
\fancyhf{}
\fancyhead[RO]{\nouppercase{\rightmark}}
\fancyhead[LE]{\nouppercase{\rightmark}}
\rfoot[]{\vspace{0.3cm}\thepage}
\lfoot[\vspace{0.3cm}\thepage]{}
%\fancyfoot[LO]{\thepage}


% Filigrane
\newcommand\BackgroundPic{%
	\put(0,0){%
		\parbox[b][\paperheight]{\paperwidth}{%
			\vfill
			\centering
			\includegraphics[width=\paperwidth,height=120mm,%
			keepaspectratio]{FichiersSource/Logos/Filigrane.eps}%
			%\vspace{1.16cm}
}}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


