import csv

import pytest

from judgepack.ingest import IngestError, load_submissions, load_winners, map_columns, redact_outcomes


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_maps_devpost_headers_and_never_maps_personal_columns():
    headers = ["Project Title", "About The Project", '"Try it out" Links', "Opt-In Prizes",
               "Submitter Email", "Team Member 1 First Name", "Submitter Screen Name", "Notes"]
    mapping, ignored, personal = map_columns(headers)
    assert mapping["title"] == "Project Title"
    assert mapping["description"] == "About The Project"
    assert mapping["links"] == '"Try it out" Links'
    assert mapping["opted_in"] == "Opt-In Prizes"
    assert set(personal) == {"Submitter Email", "Team Member 1 First Name", "Submitter Screen Name"}
    assert "Notes" in ignored
    assert not set(personal) & set(mapping.values())


def test_name_alias_does_not_grab_first_name():
    mapping, _, personal = map_columns(["First Name", "Description", "Name"])
    assert mapping["title"] == "Name"
    assert "First Name" in personal


def test_map_override_and_unknown_column():
    mapping, _, _ = map_columns(["Thing", "Words"], {"title": "Thing", "description": "Words"})
    assert mapping == {"title": "Thing", "description": "Words"}
    with pytest.raises(IngestError):
        map_columns(["Thing"], {"title": "Missing"})


@pytest.mark.parametrize(
    "text, removed",
    [
        ("Great app. Update: we won 2nd place overall!", 1),
        ("🏆 Winner of Best Use of Data", 1),
        ("We placed third. It works offline.", 1),
        ("It won't replace a doctor.", 0),
        ("We built it in the first place to learn Rust.", 0),
        ("The winner of each round gets a badge.", 0),
        ("Second place in the leaderboard is shown in silver.", 1),  # accepted false positive: cautious by design
        ("Cool Features: Score System 🏆", 0),  # decoration, not a result (seen in a real export)
        ("🏆 Our team took the prize home", 1),
    ],
)
def test_redaction(text, removed):
    _, n = redact_outcomes(text)
    assert n == removed


def test_redaction_keeps_paragraphs():
    out, n = redact_outcomes("First line.\n\nWe won the hackathon! It helps carers.\nLast line.")
    assert n == 1
    assert out == "First line.\n\nIt helps carers.\nLast line."


def test_load_submissions_end_to_end(tmp_path):
    p = write_csv(
        tmp_path / "my-hack-2024.csv",
        ["Submission Title", "About The Project", "Built With", "Submitter Email", '"Try it out" Links', "Table Number"],
        [
            ["Alpha", "Does a thing. Update: we won first place!", "Python, Flask", "a@b.c", "https://github.com/x/alpha", "2"],
            ["Beta", "Another thing that works.", "Go", "d@e.f", "", "1"],
            ["", "orphan row", "", "", "", ""],
        ],
    )
    res = load_submissions([p])
    assert [s.title for s in res.submissions] == ["Beta", "Alpha"]  # sorted by table number
    alpha = res.submissions[1]
    assert alpha.event == "My Hack 2024"
    assert alpha.redactions == 1 and "won" not in alpha.description
    assert alpha.has_repo and alpha.built_with == ["Python", "Flask"]
    assert res.files[0].personal_columns == ["Submitter Email"]
    assert res.files[0].skipped and "no project title" in res.files[0].skipped[0]
    # personal data never ends up on a submission
    assert "a@b.c" not in repr(res.submissions)


def test_missing_description_column_is_explained(tmp_path):
    p = write_csv(tmp_path / "x.csv", ["Submission Title", "Something"], [["A", "B"]])
    with pytest.raises(IngestError, match="--map"):
        load_submissions([p])


def test_winners_exact_fuzzy_and_unmatched(tmp_path):
    subs_csv = write_csv(tmp_path / "ev.csv", ["Submission Title", "About The Project"],
                         [["Spend Lens", "x"], ["Bus Buddy", "y"], ["Other", "z"]])
    subs = load_submissions([subs_csv]).submissions
    winners = write_csv(tmp_path / "w.csv", ["project", "prize"],
                        [["bus buddy", "1st"], ["SpendLens", "Best Data"], ["Nope", "2nd"]])
    report = load_winners(winners, subs)
    by_title = {s.title: s for s in subs}
    assert by_title["Bus Buddy"].prizes == ["1st"]
    assert by_title["Spend Lens"].prizes == ["Best Data"]
    assert report.matched == 2 and len(report.fuzzy) == 1 and len(report.unmatched) == 1
    assert report.winning_projects == 2
