# PSA grading standard definitions (structured)
# Source text: provided by user from PSA Grading Standards page.

psa_grade_definitions <- function(include_half_points = TRUE) {
  # Numeric grade descriptions
  numeric <- tibble::tribble(
    ~grade, ~label, ~title, ~description, ~centering_front, ~centering_back,
    10, "PSA 10", "GEM-MT", "A PSA Gem Mint 10 card is a virtually perfect card. Attributes include four perfectly sharp corners, sharp focus and full original gloss. A PSA Gem Mint 10 card must be free of staining of any kind, but an allowance may be made for a slight printing imperfection, if it doesn't impair the overall appeal of the card.", "55/45 - 60/40", "75/25",
    9, "PSA 9", "MINT", "A PSA Mint 9 is a superb condition card that exhibits only one of the following minor flaws: a very slight wax stain on reverse, a minor printing imperfection or slightly off white borders.", "60/40 or better", "90/10 or better",
    8, "PSA 8", "NM-MT", "A PSA NM-MT 8 is a super high-end card that appears Mint 9 at first glance, but upon closer inspection, the card can exhibit the following: a very slight wax stain on reverse, slightest fraying at one or two corners, a minor printing imperfection, and/or slightly off-white borders.", "65/35 or better", "90/10 or better",
    7, "PSA 7", "NM", "A PSA NM 7 is a card with just a slight surface wear visible upon close inspection. There may be slight fraying on some corners. Picture focus may be slightly out-of register. A minor printing blemish is acceptable. Slight wax staining is acceptable on the back of the card only. Most of the original gloss is retained.", "70/30 or better", "90/10 or better",
    6, "PSA 6", "EX-MT", "A PSA 6 card may have visible surface wear or a printing defect which does not detract from its overall appeal. A very light scratch may be detected only upon close inspection. Corners may have slightly graduated fraying. Picture focus may be slightly out-of-register. Card may show some loss of original gloss, may have minor wax stain on reverse, may exhibit very slight notching on edges and may also show some off-whiteness on borders.", "80/20 or better", "90/10 or better",
    5, "PSA 5", "EX", "On PSA 5 cards, very minor rounding of the corners is becoming evident. Surface wear or printing defects are more visible. There may be minor chipping on edges. Loss of original gloss will be more apparent. Focus of picture may be slightly out-of-register. Several light scratches may be visible upon close inspection, but do not detract from the appeal of the card. Card may show some off-whiteness of borders.", "85/15 or better", "90/10 or better",
    4, "PSA 4", "VG-EX", "A PSA 4 card's corners may be slightly rounded. Surface wear is noticeable but modest. The card may have light scuffing or light scratches. Some original gloss will be retained. Borders may be slightly off-white. A light crease may be visible.", "85/15 or better", "90/10 or better",
    3, "PSA 3", "VG", "A PSA 3 card reveals some rounding of the corners, though not extreme. Some surface wear will be apparent, along with possible light scuffing or light scratches. Focus may be somewhat off-register and edges may exhibit noticeable wear. Much, but not all, of the card's original gloss will be lost. Borders may be somewhat yellowed and/or discolored. A crease may be visible. Printing defects are possible. Slight stain may show on obverse and wax staining on reverse may be more prominent.", "90/10 or better", "90/10 or better",
    2, "PSA 2", "GOOD", "A PSA 2 card's corners show accelerated rounding and surface wear is starting to become obvious. A good card may have scratching, scuffing, light staining, or chipping of enamel on obverse. There may be several creases. Original gloss may be completely absent. Card may show considerable discoloration.", "90/10 or better", "90/10 or better",
    1.5, "PSA 1.5", "FR", "A PSA 1.5 card's corners will show extreme wear, possibly affecting framing of the picture. The surface of the card will show advanced stages of wear, including scuffing, scratching, pitting, chipping and staining. The picture will possibly be quite out-of-register and the borders may have become brown and dirty. The card may have one or more heavy creases. In order to achieve a Fair grade, a card must be fully intact (no missing pieces).", "90/10 or better", "90/10 or better",
    1, "PSA 1", "PR", "A PSA 1 will exhibit many of the same qualities of a PSA 1.5 but the defects may have advanced to such a serious stage that the eye appeal of the card has nearly vanished in its entirety. A Poor card may be missing one or two small pieces, exhibit major creasing that nearly breaks through all the layers of cardboard or it may contain extreme discoloration or dirtiness. It may also show noticeable warping or other destructive defects.", NA_character_, NA_character_
  )

  # Half-point grades policy (as provided)
  half_point_rule <- "Cards that exhibit high-end qualities within each particular grade, between PSA 2 and PSA 9, may achieve a half-point increase. While PSA graders will evaluate all attributes, there is a clear focus on centering when awarding half-points."

  # No Grade codes
  no_grade <- tibble::tribble(
    ~code, ~label, ~description,
    "N1", "Evidence of Trimming", "Edge appears altered by scissors, scalpel, or other tools (hooked appearance, unusually sharp edges, wavy look).",
    "N2", "Evidence of Restoration", "Paper stock appears built up (e.g., rebuilt corners).",
    "N3", "Evidence of Recoloration", "Color appears artificially improved.",
    "N4", "Questionable Authenticity", "Card appears counterfeit or autograph is not genuine.",
    "N5", "Altered Stock", "Includes paper restoration, crease/wrinkle pressing, scratch removal, enhanced gloss, or cleaning sprays/wax.",
    "N6", "Minimum Size Requirement", "Card is significantly undersized according to factory specs.",
    "N7", "Evidence of Cleaning", "Use of whitener on borders or solutions to remove stains.",
    "N8", "Miscut", "Factory cut is abnormal, causing edges to deviate from intended appearance.",
    "N9", "Don't Grade", "Used for oversized or obscure issues PSA does not currently encapsulate."
  )

  # Labels to use as folder names / class names
  # - Numeric grades: PSA_10, PSA_9, ..., PSA_1_5, PSA_1
  # - Half points: PSA_2_5 ... PSA_9_5 (optional)
  numeric <- numeric |>
    dplyr::mutate(
      class = dplyr::case_when(
        grade %% 1 == 0 ~ paste0("PSA_", as.integer(grade)),
        TRUE ~ paste0("PSA_", gsub("\\.", "_", as.character(grade)))
      )
    )

  half_points <- tibble::tibble(
    grade = c(2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
  ) |>
    dplyr::mutate(
      label = paste0("PSA ", grade),
      title = "Half-Point",
      description = half_point_rule,
      centering_front = NA_character_,
      centering_back = NA_character_,
      class = paste0("PSA_", gsub("\\.", "_", as.character(grade)))
    )

  classes <- dplyr::bind_rows(
    numeric,
    if (isTRUE(include_half_points)) half_points else NULL
  ) |>
    dplyr::select(class, grade, label, title, description, centering_front, centering_back)

  list(
    classes = classes,
    no_grade = no_grade |>
      dplyr::mutate(class = paste0("PSA_", code)) |>
      dplyr::select(class, code, label, description),
    notes = list(
      subjectivity_eye_appeal = "PSA notes that while grading is largely objective (measuring centering, finding defects), there is a subjective element. Graders reserve the right to adjust a grade based on Eye Appeal.",
      half_point_rule = half_point_rule
    )
  )
}
