# ==============================================================================
# PSA Card Grading Standards Definitions
# ==============================================================================
# Complete breakdown of PSA's card grading standards including numerical grades,
# half-point rules, and No Grade (N) definitions

# ------------------------------------------------------------------------------
# PSA Grading Standards Data Structure
# ------------------------------------------------------------------------------

psa_standards <- list(
  
  # ==========================================================================
  # NUMERICAL GRADES
  # ==========================================================================
  
  PSA_10 = list(
    grade = 10,
    name = "GEM-MT",
    full_name = "Gem Mint",
    description = paste(
      "A PSA Gem Mint 10 card is a virtually perfect card.",
      "Attributes include four perfectly sharp corners, sharp focus and full original gloss.",
      "A PSA Gem Mint 10 card must be free of staining of any kind, but an allowance may be",
      "made for a slight printing imperfection, if it doesn't impair the overall appeal of the card."
    ),
    centering = list(
      front = "55/45 - 60/40",
      back = "75/25"
    ),
    key_attributes = c(
      "Four perfectly sharp corners",
      "Sharp focus",
      "Full original gloss",
      "Free of staining",
      "Slight printing imperfection allowed if doesn't impair appeal"
    ),
    defects_allowed = c("Slight printing imperfection (if doesn't impair appeal)")
  ),
  
  PSA_9 = list(
    grade = 9,
    name = "MINT",
    full_name = "Mint",
    description = paste(
      "A PSA Mint 9 is a superb condition card that exhibits only one of the following minor flaws:",
      "a very slight wax stain on reverse, a minor printing imperfection or slightly off white borders."
    ),
    centering = list(
      front = "60/40 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Superb condition",
      "Only one minor flaw allowed"
    ),
    defects_allowed = c(
      "Very slight wax stain on reverse",
      "Minor printing imperfection",
      "Slightly off white borders"
    )
  ),
  
  PSA_8 = list(
    grade = 8,
    name = "NM-MT",
    full_name = "Near Mint-Mint",
    description = paste(
      "A PSA NM-MT 8 is a super high-end card that appears Mint 9 at first glance,",
      "but upon closer inspection, the card can exhibit the following:",
      "a very slight wax stain on reverse, slightest fraying at one or two corners,",
      "a minor printing imperfection, and/or slightly off-white borders."
    ),
    centering = list(
      front = "65/35 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Appears Mint 9 at first glance",
      "Super high-end card"
    ),
    defects_allowed = c(
      "Very slight wax stain on reverse",
      "Slightest fraying at one or two corners",
      "Minor printing imperfection",
      "Slightly off-white borders"
    )
  ),
  
  PSA_7 = list(
    grade = 7,
    name = "NM",
    full_name = "Near Mint",
    description = paste(
      "A PSA NM 7 is a card with just a slight surface wear visible upon close inspection.",
      "There may be slight fraying on some corners. Picture focus may be slightly out-of register.",
      "A minor printing blemish is acceptable. Slight wax staining is acceptable on the back of the card only.",
      "Most of the original gloss is retained."
    ),
    centering = list(
      front = "70/30 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Slight surface wear visible upon close inspection",
      "Most of the original gloss retained"
    ),
    defects_allowed = c(
      "Slight fraying on some corners",
      "Picture focus slightly out-of-register",
      "Minor printing blemish",
      "Slight wax staining on back only"
    )
  ),
  
  PSA_6 = list(
    grade = 6,
    name = "EX-MT",
    full_name = "Excellent-Mint",
    description = paste(
      "A PSA 6 card may have visible surface wear or a printing defect which does not detract",
      "from its overall appeal. A very light scratch may be detected only upon close inspection.",
      "Corners may have slightly graduated fraying. Picture focus may be slightly out-of-register.",
      "Card may show some loss of original gloss, may have minor wax stain on reverse,",
      "may exhibit very slight notching on edges and may also show some off-whiteness on borders."
    ),
    centering = list(
      front = "80/20 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Visible surface wear that doesn't detract from appeal",
      "Some loss of original gloss"
    ),
    defects_allowed = c(
      "Visible surface wear",
      "Printing defect (if doesn't detract from appeal)",
      "Very light scratch (detected upon close inspection)",
      "Slightly graduated fraying at corners",
      "Picture focus slightly out-of-register",
      "Some loss of original gloss",
      "Minor wax stain on reverse",
      "Very slight notching on edges",
      "Some off-whiteness on borders"
    )
  ),
  
  PSA_5 = list(
    grade = 5,
    name = "EX",
    full_name = "Excellent",
    description = paste(
      "On PSA 5 cards, very minor rounding of the corners is becoming evident.",
      "Surface wear or printing defects are more visible. There may be minor chipping on edges.",
      "Loss of original gloss will be more apparent. Focus of picture may be slightly out-of-register.",
      "Several light scratches may be visible upon close inspection, but do not detract from the appeal of the card.",
      "Card may show some off-whiteness of borders."
    ),
    centering = list(
      front = "85/15 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Very minor rounding of corners becoming evident",
      "Loss of original gloss more apparent",
      "Surface wear or printing defects more visible"
    ),
    defects_allowed = c(
      "Very minor rounding of corners",
      "Visible surface wear",
      "Visible printing defects",
      "Minor chipping on edges",
      "Loss of original gloss",
      "Picture focus slightly out-of-register",
      "Several light scratches (if don't detract from appeal)",
      "Some off-whiteness of borders"
    )
  ),
  
  PSA_4 = list(
    grade = 4,
    name = "VG-EX",
    full_name = "Very Good-Excellent",
    description = paste(
      "A PSA 4 card's corners may be slightly rounded. Surface wear is noticeable but modest.",
      "The card may have light scuffing or light scratches. Some original gloss will be retained.",
      "Borders may be slightly off-white. A light crease may be visible."
    ),
    centering = list(
      front = "85/15 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Corners may be slightly rounded",
      "Surface wear noticeable but modest",
      "Some original gloss retained"
    ),
    defects_allowed = c(
      "Slightly rounded corners",
      "Noticeable but modest surface wear",
      "Light scuffing",
      "Light scratches",
      "Slightly off-white borders",
      "Light crease"
    )
  ),
  
  PSA_3 = list(
    grade = 3,
    name = "VG",
    full_name = "Very Good",
    description = paste(
      "A PSA 3 card reveals some rounding of the corners, though not extreme.",
      "Some surface wear will be apparent, along with possible light scuffing or light scratches.",
      "Focus may be somewhat off-register and edges may exhibit noticeable wear.",
      "Much, but not all, of the card's original gloss will be lost.",
      "Borders may be somewhat yellowed and/or discolored. A crease may be visible.",
      "Printing defects are possible. Slight stain may show on obverse and wax staining on reverse may be more prominent."
    ),
    centering = list(
      front = "90/10 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Some rounding of corners (not extreme)",
      "Much of original gloss lost",
      "Noticeable edge wear"
    ),
    defects_allowed = c(
      "Some rounding of corners",
      "Surface wear apparent",
      "Light scuffing",
      "Light scratches",
      "Focus somewhat off-register",
      "Noticeable edge wear",
      "Much original gloss lost",
      "Somewhat yellowed/discolored borders",
      "Visible crease",
      "Printing defects",
      "Slight stain on front",
      "More prominent wax staining on reverse"
    )
  ),
  
  PSA_2 = list(
    grade = 2,
    name = "GOOD",
    full_name = "Good",
    description = paste(
      "A PSA 2 card's corners show accelerated rounding and surface wear is starting to become obvious.",
      "A good card may have scratching, scuffing, light staining, or chipping of enamel on obverse.",
      "There may be several creases. Original gloss may be completely absent.",
      "Card may show considerable discoloration."
    ),
    centering = list(
      front = "90/10 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Accelerated rounding of corners",
      "Surface wear becoming obvious",
      "Original gloss may be completely absent"
    ),
    defects_allowed = c(
      "Accelerated corner rounding",
      "Obvious surface wear",
      "Scratching",
      "Scuffing",
      "Light staining",
      "Chipping of enamel on front",
      "Several creases",
      "Complete absence of original gloss",
      "Considerable discoloration"
    )
  ),
  
  PSA_1.5 = list(
    grade = 1.5,
    name = "FR",
    full_name = "Fair",
    description = paste(
      "A PSA 1.5 card's corners will show extreme wear, possibly affecting framing of the picture.",
      "The surface of the card will show advanced stages of wear, including scuffing, scratching,",
      "pitting, chipping and staining. The picture will possibly be quite out-of-register and the",
      "borders may have become brown and dirty. The card may have one or more heavy creases.",
      "In order to achieve a Fair grade, a card must be fully intact (no missing pieces)."
    ),
    centering = list(
      front = "90/10 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Extreme corner wear",
      "Advanced surface wear",
      "Must be fully intact (no missing pieces)"
    ),
    defects_allowed = c(
      "Extreme corner wear (may affect picture framing)",
      "Advanced scuffing",
      "Advanced scratching",
      "Pitting",
      "Chipping",
      "Staining",
      "Picture quite out-of-register",
      "Brown and dirty borders",
      "One or more heavy creases"
    )
  ),
  
  PSA_1 = list(
    grade = 1,
    name = "PR",
    full_name = "Poor",
    description = paste(
      "A PSA 1 will exhibit many of the same qualities of a PSA 1.5 but the defects may have",
      "advanced to such a serious stage that the eye appeal of the card has nearly vanished",
      "in its entirety. A Poor card may be missing one or two small pieces, exhibit major creasing",
      "that nearly breaks through all the layers of cardboard or it may contain extreme discoloration",
      "or dirtiness. It may also show noticeable warping or other destructive defects."
    ),
    centering = list(
      front = "90/10 or better",
      back = "90/10 or better"
    ),
    key_attributes = c(
      "Eye appeal nearly vanished",
      "May be missing one or two small pieces",
      "Major structural damage possible"
    ),
    defects_allowed = c(
      "All defects from PSA 1.5 in advanced stage",
      "Missing one or two small pieces",
      "Major creasing (nearly breaking through cardboard)",
      "Extreme discoloration",
      "Extreme dirtiness",
      "Noticeable warping",
      "Other destructive defects"
    )
  ),
  
  # ==========================================================================
  # HALF-POINT GRADES
  # ==========================================================================
  
  half_point_rules = list(
    description = paste(
      "Cards that exhibit high-end qualities within each particular grade,",
      "between PSA 2 and PSA 9, may achieve a half-point increase.",
      "While PSA graders will evaluate all attributes, there is a clear focus on",
      "centering when awarding half-points."
    ),
    applicable_range = c(2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5),
    primary_factor = "centering",
    criteria = "High-end qualities within each particular grade"
  ),
  
  # ==========================================================================
  # NO GRADE (N) DEFINITIONS
  # ==========================================================================
  
  no_grade = list(
    
    N1 = list(
      code = "N1",
      name = "Evidence of Trimming",
      description = paste(
        "Edge appears altered by scissors, scalpel, or other tools",
        "(hooked appearance, unusually sharp edges, wavy look)."
      )
    ),
    
    N2 = list(
      code = "N2",
      name = "Evidence of Restoration",
      description = "Paper stock appears built up (e.g., rebuilt corners)."
    ),
    
    N3 = list(
      code = "N3",
      name = "Evidence of Recoloration",
      description = "Color appears artificially improved."
    ),
    
    N4 = list(
      code = "N4",
      name = "Questionable Authenticity",
      description = "Card appears counterfeit or autograph is not genuine."
    ),
    
    N5 = list(
      code = "N5",
      name = "Altered Stock",
      description = paste(
        "Includes paper restoration, crease/wrinkle pressing,",
        "scratch removal, enhanced gloss, or cleaning sprays/wax."
      )
    ),
    
    N6 = list(
      code = "N6",
      name = "Minimum Size Requirement",
      description = "Card is significantly undersized according to factory specs."
    ),
    
    N7 = list(
      code = "N7",
      name = "Evidence of Cleaning",
      description = "Use of whitener on borders or solutions to remove stains."
    ),
    
    N8 = list(
      code = "N8",
      name = "Miscut",
      description = "Factory cut is abnormal, causing edges to deviate from intended appearance."
    ),
    
    N9 = list(
      code = "N9",
      name = "Don't Grade",
      description = "Used for oversized or obscure issues PSA does not currently encapsulate."
    )
  ),
  
  # ==========================================================================
  # EYE APPEAL / SUBJECTIVITY
  # ==========================================================================
  
  eye_appeal = list(
    description = paste(
      "PSA notes that while grading is largely objective (measuring centering, finding defects),",
      "there is a subjective element. Graders reserve the right to adjust a grade based on 'Eye Appeal'."
    ),
    examples = list(
      downgrade = paste(
        "A card that technically meets centering for a PSA 9 but has a harsh color contrast",
        "making the off-centering look worse may be downgraded."
      ),
      upgrade = paste(
        "A card on the edge of centering requirements with exceptional color and corners",
        "may be granted the higher grade if the off-centering is not an 'eyesore'."
      )
    )
  )
)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

#' Get grade information by grade value
#' @param grade_value Numeric grade value (1-10)
#' @return List containing grade information
get_grade_info <- function(grade_value) {
  grade_key <- paste0("PSA_", grade_value)
  if (grade_key %in% names(psa_standards)) {
    return(psa_standards[[grade_key]])
  } else {
    stop(paste("Invalid grade value:", grade_value))
  }
}

#' Get all defects allowed for a grade
#' @param grade_value Numeric grade value (1-10)
#' @return Character vector of allowed defects
get_allowed_defects <- function(grade_value) {
  info <- get_grade_info(grade_value)
  return(info$defects_allowed)
}

#' Get centering requirements for a grade
#' @param grade_value Numeric grade value (1-10)
#' @return List with front and back centering requirements
get_centering_requirements <- function(grade_value) {
  info <- get_grade_info(grade_value)
  return(info$centering)
}

#' Get no-grade information by code
#' @param code No-grade code (N1-N9)
#' @return List containing no-grade information
get_no_grade_info <- function(code) {
  if (code %in% names(psa_standards$no_grade)) {
    return(psa_standards$no_grade[[code]])
  } else {
    stop(paste("Invalid no-grade code:", code))
  }
}

#' Print summary of all PSA grades
print_grade_summary <- function() {
  cat("=== PSA Card Grading Standards Summary ===\n\n")
  
  grades <- c("PSA_10", "PSA_9", "PSA_8", "PSA_7", "PSA_6", 
              "PSA_5", "PSA_4", "PSA_3", "PSA_2", "PSA_1.5", "PSA_1")
  
  for (grade_key in grades) {
    grade <- psa_standards[[grade_key]]
    cat(sprintf("PSA %s (%s) - %s\n", grade$grade, grade$name, grade$full_name))
    cat(sprintf("  Centering: Front %s | Back %s\n", 
                grade$centering$front, grade$centering$back))
    cat("\n")
  }
  
  cat("\n=== No Grade Codes ===\n\n")
  for (code in names(psa_standards$no_grade)) {
    ng <- psa_standards$no_grade[[code]]
    cat(sprintf("%s: %s\n", ng$code, ng$name))
  }
}

#' Create a data frame of centering requirements
#' @return Data frame with centering requirements for each grade
get_centering_table <- function() {
  grades <- c("PSA_10", "PSA_9", "PSA_8", "PSA_7", "PSA_6", 
              "PSA_5", "PSA_4", "PSA_3", "PSA_2", "PSA_1.5", "PSA_1")
  
  data.frame(
    grade = sapply(grades, function(g) psa_standards[[g]]$grade),
    name = sapply(grades, function(g) psa_standards[[g]]$name),
    front_centering = sapply(grades, function(g) psa_standards[[g]]$centering$front),
    back_centering = sapply(grades, function(g) psa_standards[[g]]$centering$back),
    row.names = NULL
  )
}
