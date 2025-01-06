from otree.api import *

doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'demographics'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    years = range(1920, 2005)
    list_of_education = [
        'No School Graduation',
        'High School Graduation',
        "Bachelor's degree or equivalent",
        "Master's degree or equivalent",
        'Doctorate'
    ]


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


def likertScale(label, low, high, n, blank = False):
    if low != '':
        CHOICES = [(0, str(0) + " (" + low + ")")]
    else:
        CHOICES = [(0, str(0))]
    for x in range(1, n):
        CHOICES = CHOICES + [(x, str(x))]
    if high != '':
        CHOICES = CHOICES + [(n, str(n) + " (" + high + ")")]
    else:
        CHOICES = CHOICES + [(n, str(n))]
    return models.IntegerField(
        choices=CHOICES,
        label=label,
        widget=widgets.RadioSelectHorizontal,
        blank=blank,
    )

class Player(BasePlayer):
    yearOfBirth = models.IntegerField(choices=C.years, label="What is your year of birth?")
    male = models.IntegerField(widget=widgets.RadioSelectHorizontal, choices=[[0, 'female'], [1, 'male'], [99, 'diverse'],
                                                                                [98, 'Do not want to say'], ],
                               label="What is your gender?")
    education = models.StringField(choices=C.list_of_education,
                                   label="What is your highest level of education?")
    country = models.StringField(label="In which country do you live?")

    politics =likertScale(
        'On a scale of 0 to 10, on which 0 means far left and 10 means far right, where do you place your political views?',
        '', '', 10)
    finishexptime = models.IntegerField(initial=0) #end of experiment
    startdemotime = models.IntegerField(initial=0) #start of demographics survey


    HintHelpful = likertScale(
        'Did you think the messages were helpful? Please use a scale from 0 to 10, where a 0 means you found them "not helpful at all" and a 10 means you found them "very helpful".',
        '', '', 10)

    HintMisleading = likertScale(
        'Did you think the messages were misleading? Please use a scale from 0 to 10, where a 0 means you believe they were "not misleading at all" and a 10 means you believe they were "very misleading".',
        '', '', 10)

    TimeSurvey = likertScale(
        'How willing are you to give up something today in order to get more of it in the future? Please use a scale from 0 to 10, where a 0 means you are "completely unwilling to give up something today for more tomorrow" and a 10 means you are "very willing to give up something today for more tomorrow".',
        '', '', 10)
    RiskSurvey = likertScale(
        'How willing are you in general to take risk? Please use a scale from 0 to 10, where a 0 means you are "completely unwilling to take risks" and a 10 means you are "very willing to take risks".',
        '', '', 10)
    AltruismSurvey = likertScale(
        'How do you assess your willingness to share with others without expecting anything in return when it comes to charity? Please use a scale from 0 to 10, where 0 means you are "completely unwilling to share" and a 10 means you are" very willing to share”.',
        '', '', 10)

    CR1 = models.IntegerField(label='If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 days, how many days would it take them to drink one barrel of water together?')
    CR2 = models.IntegerField(label='A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it finally for $90. How much profit has he made, in dollars?')
    CR3 = models.IntegerField(label='Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are in the class?')

# PAGES
class Demographics(Page):
    form_model = 'player'
    form_fields = ['yearOfBirth', 'male', 'country', 'education', 'politics']

    def vars_for_template(player: Player):
        import time
        player.startdemotime = int(time.time())
        return dict()

class Risk_Narratives(Page):
    form_model = 'player'
    form_fields = ['RiskSurvey', 'HintHelpful', 'HintMisleading', 'CR1', 'CR2', 'CR3']

class ResultsWaitPage(WaitPage):
    pass

class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        import time
        player.finishexptime = int(time.time())

        combined_link = player.subsession.session.config['limesurvey_link'] + f'&PAYMENTCODE={player.participant.label}'
        return dict(combined_link=combined_link)

page_sequence = [Demographics, Risk_Narratives, Results]
