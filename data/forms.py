from wtforms import Form, StringField, SelectField

class BookSearchForm(Form):
    choices = [("Author", "Author"), ("Book", "Book")]
    select = SelectField("Search for a book: ", choices=choices)
    search = StringField("")