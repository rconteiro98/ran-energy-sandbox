---
name: geo-module
description: Build and refine the UCL Bloomsbury study-area and candidate-site loading module.
---

# Geo Module Skill

## Purpose
This module defines the UCL Bloomsbury study area and loads approximate candidate nearby radio sites for research prototyping.

## Scope
Work only on:
- `src/geo/__init__.py`
- `src/geo/campus_area.py`
- `data/raw/candidate_sites.csv`

## Requirements
- Define an approximate UCL Bloomsbury campus center point.
- Define a study radius in meters.
- Load candidate sites from CSV.
- Compute distance from campus center.
- Filter sites within the study radius.
- Return clean pandas DataFrames.
- Include a small synthetic `candidate_sites.csv`.
- Clearly document that the site data is approximate and for prototyping only.

## Constraints
- Keep implementation simple and interview-ready.
- Use standard Python plus pandas.
- Avoid unnecessary geospatial dependencies for the first version.
- Use a simple distance approximation or haversine formula.

## Definition of done
- The module runs independently.
- Running the module prints filtered study-area sites.
- Code is readable and minimal.